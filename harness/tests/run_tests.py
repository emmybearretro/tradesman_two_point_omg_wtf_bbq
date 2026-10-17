"""End-to-end tests against a local OpenAI-compatible stub. Run: python3 tests/run_tests.py"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness.config import Config, JobCfg, ProviderCfg, expand, load  # noqa: E402
from harness.pipeline import (Cache, Harness, JobError, Result,  # noqa: E402
                              extract_json, load_items, validate)
from harness.provider import Provider, ProviderError  # noqa: E402
from harness.registry import Registry  # noqa: E402

CALLS: list[dict] = []
LOCK = threading.Lock()


class Stub(BaseHTTPRequestHandler):
    """Serves models, chat, streaming, and scripted failures keyed off model id."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = json.dumps(obj).encode() if not isinstance(obj, bytes) else obj
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send({"object": "list", "data": [
                {"id": "stub-a"}, {"id": "stub-b"}, {"id": "stub-strict-only"}]})
        return self._send({"error": "no route"}, 404)

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        model = req.get("model", "?")
        prompt = " ".join(m.get("content", "") for m in req.get("messages", []))
        with LOCK:
            CALLS.append({"model": model, "auth": self.headers.get("authorization"),
                          "prompt": prompt, "stream": req.get("stream"),
                          "response_format": req.get("response_format"),
                          "temperature": req.get("temperature")})

        if model == "stub-400":
            return self._send({"error": {"message": "not loaded"}}, 400)
        if model == "stub-429-then-ok":
            with LOCK:
                retries = sum(1 for c in CALLS if c["model"] == model)
            if retries < 3:
                return self._send({"error": {"message": "slow down"}}, 429)
            model_out = "stub-a"
        elif model == "stub-chatty-json":
            return self._send({"model": model, "choices": [{"message": {"content":
                "Here:\n```json\n{\"a\": 1}\n```\nDone"}, "finish_reason": "stop"}]})
        elif model == "stub-bad-json":
            return self._send({"model": model, "choices": [{"message": {"content":
                "no json here"}, "finish_reason": "stop"}]})
        elif model == "stub-stream":
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for piece in ["hel", "lo ", "world"]:
                self.wfile.write(f"data: {json.dumps({'model': model, 'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        else:
            model_out = model

        fmt = (req.get("response_format") or {}).get("type")
        if fmt in ("json_object", "json_schema"):
            content = json.dumps({"category": "bug", "severity": "high", "echo": prompt})
        else:
            content = f"reply::{prompt}"
        self._send({"model": model_out, "choices": [{"message": {"content": content},
                    "finish_reason": "stop"}], "usage": {"total_tokens": 9}})


SERVER: ThreadingHTTPServer | None = None
BASE = ""


def setUpModule():
    global SERVER, BASE
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    BASE = f"http://127.0.0.1:{SERVER.server_address[1]}/v1"
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def tearDownModule():
    if SERVER:
        SERVER.shutdown()
        SERVER.server_close()


def make_config(tmp: str) -> Config:
    """Write a config pointing at the stub; __BASE__/__TMP__ keep TOML braces literal."""
    toml = """
cache_dir = "__TMP__/cache"

[providers.stub]
base_url = "__BASE__"
default = true
max_retries = 2

[providers.strict]
base_url = "__BASE__"
strict = true
api_key = "sk-secret"

[providers.hints]
base_url = "__BASE__"
models = ["declared-only"]

[aliases]
pair = ["stub/stub-a", "stub/stub-b"]
cycle-a = ["cycle-b"]
cycle-b = ["cycle-a"]

[jobs.plain]
model = "pair"
system = "sys"

[jobs.typed]
model = "stub/stub-a"
format = "schema"
template = "T: {input}"
schema = { type = "object", required = ["category", "severity"],
  properties = { category = { type = "string", enum = ["bug", "feature"] },
                 severity = { type = "string", enum = ["low", "med", "high"] } } }

[jobs.fallback]
model = ["stub/stub-400", "stub/stub-b"]

[jobs.retried]
model = "stub/stub-429-then-ok"

[jobs.chatty]
model = "stub/stub-chatty-json"
format = "json"

[jobs.unparseable]
model = "stub/stub-bad-json"
format = "json"

[jobs.streamy]
model = "stub/stub-stream"

[jobs.strictjob]
model = "strict/stub-strict-only"

[jobs.templated]
model = "stub/stub-a"
template = "field={field} input={input}"

[jobs.broken-template]
model = "stub/stub-a"
template = "nope={missing}"

[pipelines.two-stage]
stages = [{ job = "typed" }, { job = "plain", input = "json" }]

[pipelines.field-fed]
stages = [{ job = "plain", input = "field:body", model = "stub/stub-b" }]

[pipelines.failfast]
fail_fast = true
stages = [{ job = "unparseable" }, { job = "plain" }]
"""
    path = os.path.join(tmp, "config.toml")
    with open(path, "w") as f:
        f.write(toml.replace("__BASE__", BASE).replace("__TMP__", tmp))
    return load(path)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="harness-test-")
        self.cfg = make_config(self.tmp)
        self.h = Harness(self.cfg)
        self.h.registry.refresh(ttl=0)
        CALLS.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Discovery(Base):
    def test_lists_served_models(self):
        self.h.registry.refresh(ttl=0)
        self.assertEqual(self.h.registry.providers["stub"].listed,
                         ["stub-a", "stub-b", "stub-strict-only"])

    def test_config_models_survive_discovery(self):
        """Declared ids stay resolvable even when the endpoint omits them."""
        self.h.registry.refresh(ttl=0)
        listed = self.h.registry.providers["hints"].listed
        self.assertIn("declared-only", listed)
        self.assertIn("stub-a", listed)          # discovered ids kept too
        target = self.h.registry.resolve(["hints/declared-only"])[0]
        self.assertTrue(target.listed)

    def test_unreachable_provider_recorded(self):
        self.cfg.providers["dead"] = ProviderCfg(name="dead", base_url="http://127.0.0.1:1/v1")
        reg = Registry(self.cfg)
        reg.refresh(ttl=0)
        self.assertIn("dead", reg.discovery_errors())

    def test_new_model_appears_after_refresh(self):
        """Models load over time: a bare id is unlisted, then listed after refresh."""
        reg = Registry(self.cfg)
        reg.refresh(ttl=0)
        self.assertFalse(reg.known("stub", "brand-new"))
        target = reg.resolve(["brand-new"])[0]
        self.assertEqual((target.provider, target.listed), ("stub", False))


class Resolution(Base):
    def test_alias_expands_to_chain(self):
        self.assertEqual([str(t) for t in self.h.registry.resolve("pair")],
                         ["stub/stub-a", "stub/stub-b"])

    def test_alias_of_alias(self):
        self.cfg.aliases["outer"] = ["pair"]
        reg = Registry(self.cfg)
        self.assertEqual(len(reg.resolve("outer")), 2)

    def test_bare_id_finds_provider_serving_it(self):
        self.assertEqual(str(self.h.registry.resolve("stub-a")[0]), "stub/stub-a")

    def test_fuzzy_prefix_match(self):
        self.assertEqual(str(self.h.registry.resolve("stub-")[0]), "stub/stub-a")

    def test_strict_provider_rejects_unlisted(self):
        with self.assertRaises(ValueError) as ctx:
            self.h.registry.resolve(["strict/nope"])
        self.assertIn("strict is strict", str(ctx.exception))

    def test_strict_member_skipped_rest_of_chain_kept(self):
        chain = self.h.registry.resolve(["strict/nope", "stub/stub-a"])
        self.assertEqual([str(t) for t in chain], ["stub/stub-a"])
        self.assertEqual(len(chain.notes), 1)

    def test_alias_cycle_reported_not_fatal(self):
        chain = self.h.registry.resolve(["cycle-a", "stub/stub-a"])
        self.assertEqual([str(t) for t in chain], ["stub/stub-a"])
        self.assertTrue(any("cycle" in n for n in chain.notes))

    def test_unresolvable_raises(self):
        with self.assertRaises(ValueError):
            self.h.registry.resolve(["strict/nope"])


class Chat(Base):
    def test_plain_reply(self):
        r = self.h.run("plain", "hello", use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "reply::sys hello")
        self.assertEqual(r.target, "stub/stub-a")

    def test_schema_job_sends_structured_output_request(self):
        r = self.h.run("typed", {"input": "boom"}, use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.parsed["category"], "bug")
        fmt = CALLS[-1]["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["name"], "typed")
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertEqual(fmt["json_schema"]["schema"]["required"],
                         ["category", "severity"])

    def test_template_uses_item_fields(self):
        r = self.h.run("templated", {"field": "F", "input": "I"}, use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertIn("field=F input=I", r.text)

    def test_template_missing_field_is_job_error(self):
        with self.assertRaises(JobError):
            self.h.run("broken-template", "x", use_cache=False)

    def test_messages_passthrough(self):
        r = self.h.run("plain", {"messages": [{"role": "user", "content": "raw"}]},
                       use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "reply::raw")

    def test_api_key_sent_for_keyed_provider(self):
        r = self.h.run("strictjob", "x", use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(CALLS[-1]["auth"], "Bearer sk-secret")

    def test_no_auth_header_without_key(self):
        self.h.run("plain", "x", use_cache=False)
        self.assertIsNone(CALLS[-1]["auth"])

    def test_streaming_assembles_deltas(self):
        seen = []
        r = self.h.run("streamy", "go", stream=True, on_delta=seen.append, use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "hello world")
        self.assertEqual("".join(seen), "hello world")

    def test_429_retried_then_succeeds(self):
        r = self.h.run("retried", "x", use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertGreater(r.attempts, 1)

    def test_400_not_retried_falls_back(self):
        r = self.h.run("fallback", "x", use_cache=False)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.target, "stub/stub-b")
        self.assertEqual([t.split(" ")[0] for t in r.tried], ["stub/stub-400", "stub/stub-b"])
        self.assertEqual(sum(1 for c in CALLS if c["model"] == "stub-400"), 1)

    def test_all_models_fail_reports_chain(self):
        self.cfg.jobs["plain"].model = ["stub/stub-400"]
        r = self.h.run("plain", "x", use_cache=False)
        self.assertFalse(r.ok)
        self.assertIn("HTTP 400", r.error)

    def test_unparseable_json_fails_with_schema_note(self):
        r = self.h.run("unparseable", "x", use_cache=False)
        self.assertFalse(r.ok)
        self.assertIn("no JSON value", r.error)


class Parsing(unittest.TestCase):
    def test_extract_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_extract_fenced_json(self):
        self.assertEqual(extract_json('prose\n```json\n{"a": 1}\n```\ntail'), {"a": 1})

    def test_extract_embedded_object(self):
        self.assertEqual(extract_json('x {"a": {"b": 2}} y'), {"a": {"b": 2}})

    def test_extract_array(self):
        self.assertEqual(extract_json('out: [1, 2]'), [1, 2])

    def test_extract_string_with_braces(self):
        self.assertEqual(extract_json('{"a": "} not a close"}'), {"a": "} not a close"})

    def test_extract_nothing_raises(self):
        with self.assertRaises(ValueError):
            extract_json("no json at all")

    def test_validate_ok(self):
        self.assertEqual(validate({"a": "x"}, {"type": "object", "required": ["a"],
                                               "properties": {"a": {"type": "string"}}}), [])

    def test_validate_missing_required(self):
        self.assertIn("missing required 'a'",
                      " ".join(validate({}, {"required": ["a"]})))

    def test_validate_enum(self):
        self.assertIn("not in enum", " ".join(
            validate("z", {"type": "string", "enum": ["a", "b"]})))

    def test_validate_nested_array_items(self):
        errs = validate([{"n": 1}, {"n": "x"}],
                        {"type": "array", "items": {"properties": {"n": {"type": "integer"}}}})
        self.assertEqual(len(errs), 1)
        self.assertIn("[1]", errs[0])

    def test_validate_bool_is_not_integer(self):
        self.assertTrue(validate(True, {"type": "integer"}))

    def test_validate_minimum(self):
        self.assertIn("minimum", " ".join(validate(-1, {"type": "integer", "minimum": 0})))


class CacheTests(Base):
    def test_second_run_served_from_cache(self):
        first = self.h.run("plain", "cached-item")
        self.assertTrue(first.ok, first.error)
        self.assertFalse(first.cached)
        second = self.h.run("plain", "cached-item")
        self.assertTrue(second.cached)
        self.assertEqual(second.text, first.text)
        self.assertEqual(len([c for c in CALLS if c["prompt"].endswith("cached-item")]), 1)

    def test_no_cache_bypasses(self):
        self.h.run("plain", "x")
        r = self.h.run("plain", "x", use_cache=False)
        self.assertFalse(r.cached)
        self.assertEqual(len(CALLS), 2)

    def test_cache_key_includes_model_chain(self):
        self.h.run("plain", "x")
        self.cfg.jobs["plain"].model = ["stub/stub-b"]
        r = self.h.run("plain", "x")
        self.assertFalse(r.cached, "chain change must not reuse a cached reply")

    def test_cache_key_includes_schema(self):
        self.h.run("typed", {"input": "x"})
        self.cfg.jobs["typed"].schema = {"type": "object", "required": ["other"]}
        r = self.h.run("typed", {"input": "x"})
        self.assertFalse(r.cached)

    def test_cache_roundtrip_atomic(self):
        c = Cache(os.path.join(self.tmp, "c2"))
        c.put("k1", {"text": "v"})
        self.assertEqual(c.get("k1"), {"text": "v"})
        self.assertIsNone(c.get("missing"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "c2", "k1.json.tmp")))


class FanOut(Base):
    def test_results_keep_input_order(self):
        items = [f"item-{i}" for i in range(8)]
        self.cfg.jobs["plain"].max_workers = 8
        results = self.h.run_job("plain", items, use_cache=False)
        self.assertEqual([r.index for r in results], list(range(8)))
        for i, r in enumerate(results):
            self.assertTrue(r.ok, r.error)
            self.assertEqual(r.text, f"reply::sys item-{i}")

    def test_failures_do_not_abort_batch(self):
        self.cfg.jobs["plain"].model = ["stub/stub-400"]
        results = self.h.run_job("plain", ["a", "b"], use_cache=False)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not r.ok for r in results))

    def test_on_result_sees_every_item(self):
        seen = []
        self.h.run_job("plain", ["a", "b", "c"], use_cache=False,
                       on_result=lambda r: seen.append(r.index))
        self.assertEqual(sorted(seen), [0, 1, 2])

    def test_load_items_formats(self):
        lines = os.path.join(self.tmp, "i.txt")
        with open(lines, "w") as f:
            f.write("one\n\ntwo\n")
        self.assertEqual(load_items(lines), ["one", "two"])

        js = os.path.join(self.tmp, "i.json")
        with open(js, "w") as f:
            json.dump([{"key": "a"}, {"key": "b"}], f)
        self.assertEqual([d["key"] for d in load_items(js)], ["a", "b"])

        jl = os.path.join(self.tmp, "i.jsonl")
        with open(jl, "w") as f:
            f.write('{"key": "x"}\n\n{"key": "y"}\n')
        self.assertEqual([d["key"] for d in load_items(jl)], ["x", "y"])


class Pipelines(Base):
    def test_stage_output_feeds_next_stage(self):
        rows = self.h.run_pipeline("two-stage", [{"key": "k", "input": "boom"}],
                                   use_cache=False)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["ok"], row["error"])
        self.assertEqual([s["job"] for s in row["stages"]], ["typed", "plain"])
        self.assertEqual(row["stages"][0]["parsed"]["category"], "bug")
        self.assertIn('"category": "bug"', row["stages"][1]["text"])
        self.assertEqual(row["model"], "stub/stub-a")

    def test_field_selector_reads_original_item(self):
        rows = self.h.run_pipeline("field-fed", [{"key": "k", "body": "raw body"}],
                                   use_cache=False)
        self.assertTrue(rows[0]["ok"], rows[0]["error"])
        self.assertIn("raw body", rows[0]["text"])
        self.assertEqual(rows[0]["model"], "stub/stub-b")

    def test_fail_fast_stops_after_stage_one(self):
        rows = self.h.run_pipeline("failfast", [{"key": "k", "input": "x"}],
                                   use_cache=False)
        self.assertFalse(rows[0]["ok"])
        self.assertEqual([s["job"] for s in rows[0]["stages"]], ["unparseable"])

    def test_pipeline_parallel_items(self):
        rows = self.h.run_pipeline("field-fed",
                                   [{"key": f"k{i}", "body": f"body{i}"} for i in range(4)],
                                   use_cache=False)
        self.assertEqual([r["key"] for r in rows], ["k0", "k1", "k2", "k3"])
        for i, row in enumerate(rows):
            self.assertIn(f"body{i}", row["text"])

    def test_unknown_pipeline_raises(self):
        with self.assertRaises(JobError):
            self.h.run_pipeline("nope", ["x"])


class ConfigTests(unittest.TestCase):
    def test_env_expansion_with_default(self):
        os.environ["HARNESS_TEST_URL"] = "http://env:1/v1"
        self.assertEqual(expand("${HARNESS_TEST_URL}"), "http://env:1/v1")
        self.assertEqual(expand("${HARNESS_TEST_UNSET:-http://fallback/v1}"),
                         "http://fallback/v1")
        self.assertEqual(expand({"k": ["${HARNESS_TEST_URL}"]}), {"k": ["http://env:1/v1"]})
        del os.environ["HARNESS_TEST_URL"]

    def test_missing_config_path_exits(self):
        with self.assertRaises(SystemExit):
            load("/nonexistent/config.toml")

    def test_no_config_found_exits(self):
        cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.environ.pop("HARNESS_CONFIG", None)
            with self.assertRaises(SystemExit):
                load(None)
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pipeline_with_unknown_job_exits(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "c.toml")
            with open(path, "w") as f:
                f.write('[providers.p]\nbase_url = "http://x/v1"\n\n'
                        '[pipelines.p1]\nstages = [{ job = "ghost" }]\n')
            with self.assertRaises(SystemExit):
                load(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_default_provider_prefers_flag(self):
        cfg = Config(path="p", providers={
            "a": ProviderCfg(name="a", base_url="x"),
            "b": ProviderCfg(name="b", base_url="y", default=True)},
            aliases={}, jobs={}, pipelines={}, cache_dir="c")
        self.assertEqual(cfg.default_provider, "b")


class ProviderUnit(unittest.TestCase):
    def test_url_join_normalizes_slashes(self):
        p = Provider(name="p", base_url="http://h:1/v1")
        self.assertEqual(p.url("chat/completions"), "http://h:1/v1/chat/completions")

    def test_extra_body_params_merged_into_request(self):
        p = Provider(name="p", base_url=BASE, extra={"temperature": 0.1})
        res = p.chat([{"role": "user", "content": "x"}], "stub-a")
        self.assertTrue(res.text.startswith("reply::"))
        self.assertEqual(CALLS[-1]["temperature"], 0.1)

    def test_per_call_params_override_provider_extra(self):
        p = Provider(name="p", base_url=BASE, extra={"temperature": 0.1})
        p.chat([{"role": "user", "content": "x"}], "stub-a", params={"temperature": 0.9})
        self.assertEqual(CALLS[-1]["temperature"], 0.9)

    def test_unreachable_is_retryable_error(self):
        p = Provider(name="p", base_url="http://127.0.0.1:1/v1", max_retries=0)
        with self.assertRaises(ProviderError) as ctx:
            p.chat([{"role": "user", "content": "x"}], "m")
        self.assertTrue(ctx.exception.retryable)


class CLI(Base):
    def run_cli(self, argv):
        from harness.cli import main
        return main(argv)

    def rows(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    def test_broken_pipe_exits_quietly(self):
        """`| head` must not print a traceback."""
        cmd = [sys.executable, "-m", "harness", "-c", self.cfg.path, "jobs"]
        proc = subprocess.run(" ".join(shlex.quote(a) for a in cmd) + " | head -1",
                              shell=True, cwd=ROOT, capture_output=True, text=True)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("plain", proc.stdout)

    def test_models_exit_zero_when_all_reachable(self):
        self.assertEqual(self.run_cli(["-c", self.cfg.path, "models"]), 0)

    def test_ping_exit_one_when_provider_dead(self):
        self.cfg.providers["dead"] = ProviderCfg(name="dead", base_url="http://127.0.0.1:1/v1")
        path = os.path.join(self.tmp, "withdead.toml")
        with open(path, "w") as f:
            f.write(f'[providers.stub]\nbase_url = "{BASE}"\n\n'
                    f'[providers.dead]\nbase_url = "http://127.0.0.1:1/v1"\n')
        self.assertEqual(self.run_cli(["-c", path, "ping"]), 1)

    def test_run_writes_jsonl_and_reports(self):
        out = os.path.join(self.tmp, "out.jsonl")
        code = self.run_cli(["-c", self.cfg.path, "run", "plain", "-p", "one",
                             "-p", "two", "-o", out, "--no-cache", "-q"])
        self.assertEqual(code, 0)
        rows = self.rows(out)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["ok"] for r in rows))

    def test_run_creates_missing_output_directory(self):
        out = os.path.join(self.tmp, "nested", "deep", "out.jsonl")
        code = self.run_cli(["-c", self.cfg.path, "run", "plain", "-p", "one",
                             "-o", out, "--no-cache", "-q"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(out))

    def test_run_reads_jsonl_from_stdin(self):
        out = os.path.join(self.tmp, "stdin.jsonl")
        payload = '{"key":"a","input":"one"}\n{"key":"b","input":"two"}\n'
        real = sys.stdin
        sys.stdin = io.StringIO(payload)
        try:
            code = self.run_cli(["-c", self.cfg.path, "run", "plain", "-i", "-",
                                 "-o", out, "--no-cache", "-q"])
        finally:
            sys.stdin = real
        self.assertEqual(code, 0)
        rows = self.rows(out)
        self.assertEqual(sorted(r["key"] for r in rows), ["a", "b"])

    def test_run_reads_plain_lines_from_stdin(self):
        out = os.path.join(self.tmp, "stdin2.jsonl")
        real = sys.stdin
        sys.stdin = io.StringIO("alpha\nbeta\n")
        try:
            code = self.run_cli(["-c", self.cfg.path, "run", "plain", "-i", "-",
                                 "-o", out, "--no-cache", "-q"])
        finally:
            sys.stdin = real
        self.assertEqual(code, 0)
        self.assertEqual(len(self.rows(out)), 2)

    def test_run_exit_one_on_failure(self):
        out = os.path.join(self.tmp, "fail.jsonl")
        code = self.run_cli(["-c", self.cfg.path, "-m", "stub/stub-400", "run", "plain",
                             "-p", "one", "-o", out, "--no-cache", "-q"])
        self.assertEqual(code, 1)

    def test_run_unknown_job_exit_two(self):
        self.assertEqual(self.run_cli(["-c", self.cfg.path, "run", "ghost", "-p", "x"]), 2)

    def test_ask_needs_job_or_model(self):
        self.assertEqual(self.run_cli(["-c", self.cfg.path, "ask", "hi"]), 2)

    def test_ask_with_model_override(self):
        self.assertEqual(self.run_cli(["-c", self.cfg.path, "ask", "-m", "stub/stub-a",
                                       "--no-cache", "hi"]), 0)

    def test_pipe_writes_rows(self):
        items = os.path.join(self.tmp, "in.jsonl")
        with open(items, "w") as f:
            f.write('{"key":"k","input":"boom"}\n')
        out = os.path.join(self.tmp, "pipe.jsonl")
        code = self.run_cli(["-c", self.cfg.path, "pipe", "two-stage", "-i", items,
                             "-o", out, "--no-cache", "-q"])
        self.assertEqual(code, 0)
        row = self.rows(out)[0]
        self.assertEqual([s["job"] for s in row["stages"]], ["typed", "plain"])

    def test_pipe_exit_one_on_failure(self):
        items = os.path.join(self.tmp, "in2.jsonl")
        with open(items, "w") as f:
            f.write('{"key":"k","input":"x"}\n')
        out = os.path.join(self.tmp, "pipe2.jsonl")
        code = self.run_cli(["-c", self.cfg.path, "pipe", "failfast", "-i", items,
                             "-o", out, "--no-cache", "-q"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
