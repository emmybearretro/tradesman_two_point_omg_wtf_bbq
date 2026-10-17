"""Jobs: prompt building, model fallback, JSON/schema parsing, cache, parallel fan-out."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, JobCfg
from .provider import ChatResult, ProviderError
from .registry import Registry

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
JSON_SPAN_RE = re.compile(r"[\{\[]", )


class JobError(Exception):
    pass


@dataclass
class Result:
    job: str
    index: int
    key: str | None
    ok: bool
    text: str = ""
    parsed: object = None
    target: str | None = None
    provider: str | None = None
    model: str | None = None
    attempts: int = 0
    latency: float = 0.0
    usage: dict = field(default_factory=dict)
    cached: bool = False
    tried: list[str] = field(default_factory=list)
    error: str | None = None

    def record(self) -> dict:
        return {
            "job": self.job, "index": self.index, "key": self.key, "ok": self.ok,
            "model": self.target, "attempts": self.attempts, "latency": round(self.latency, 3),
            "cached": self.cached, "usage": self.usage, "error": self.error,
            "text": self.text, "parsed": self.parsed,
        }


# ---- parsing -------------------------------------------------------------

def extract_json(text: str):
    """Pull a JSON value out of a chatty model reply."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for cand in (m.strip() for m in FENCE_RE.findall(text)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    starts = [m.start() for m in JSON_SPAN_RE.finditer(text)]
    for s in starts:
        opener = text[s]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = esc = False
        for i in range(s, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON value found in reply")


def validate(value, schema, path="$") -> list[str]:
    """Small JSON-Schema subset: type, required, properties, items, enum, minimum/maximum."""
    errs: list[str] = []
    if not isinstance(schema, dict):
        return errs
    t = schema.get("type")
    types = {
        "object": dict, "array": list, "string": str,
        "integer": int, "number": (int, float), "boolean": bool, "null": type(None),
    }
    if t and t in types:
        if t in ("integer", "number") and isinstance(value, bool):
            errs.append(f"{path}: expected {t}")
            return errs
        if not isinstance(value, types[t]):
            errs.append(f"{path}: expected {t}, got {type(value).__name__}")
            return errs
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}: missing required '{req}'")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                errs.extend(validate(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errs.extend(validate(item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    return errs


# ---- cache ---------------------------------------------------------------

class Cache:
    def __init__(self, directory: str):
        self.dir = Path(directory)

    def key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def get(self, key: str):
        p = self.dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f"{key}.json.tmp"
            tmp.write_text(json.dumps(value, ensure_ascii=False), "utf-8")
            os.replace(tmp, self.dir / f"{key}.json")
        except OSError:
            pass


def _schema_name(job_name: str) -> str:
    """OpenAI-compatible response_format names: [a-zA-Z0-9_-], <= 64 chars."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", job_name) or "job"
    return safe[:64]

# ---- harness -------------------------------------------------------------

class Harness:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.registry = Registry(cfg)
        self.cache = Cache(cfg.cache_dir)

    # ---- prompt ----------------------------------------------------------
    def messages(self, job: JobCfg, item) -> list[dict]:
        if isinstance(item, dict) and "messages" in item:
            return list(item["messages"])
        body = _item_text(item)
        prompt = body
        if job.template:
            fields = {k: v for k, v in (item.items() if isinstance(item, dict) else [])
                      if k != "input" and isinstance(v, str)}
            try:
                prompt = job.template.format(input=body, **fields)
            except (KeyError, IndexError) as e:
                raise JobError(f"job '{job.name}' template {job.template!r}: "
                               f"no field {e} in item") from e
        msgs = []
        if job.system:
            msgs.append({"role": "system", "content": job.system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _params(self, job: JobCfg, prov) -> dict:
        """Sampling params plus the response_format shape this provider accepts.

        `prompt` mode sends nothing on the wire: it is for endpoints that reject
        structured-output fields outright.
        """
        params = dict(job.params)
        if job.format == "schema" and job.schema:
            if prov.schema_mode == "json_schema":
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": _schema_name(job.name),
                                    "strict": True,
                                    "schema": job.schema},
                }
            elif prov.schema_mode == "json_object":
                params["response_format"] = {"type": "json_object"}
        elif job.format == "json":
            params["response_format"] = {"type": "json_object"}
        return params

    def _schema_hint(self, job: JobCfg, prov) -> str:
        """Schema text for providers that cannot enforce one on the wire."""
        if job.format == "schema" and job.schema and prov.schema_mode == "prompt":
            return ("\nReply with a single JSON object matching this schema, "
                    "no prose:\n" + json.dumps(job.schema))
        return ""

    def _parse(self, job: JobCfg, text: str):
        if job.format == "text":
            return text, None
        value = extract_json(text)
        if job.format == "schema" and job.schema:
            errs = validate(value, job.schema)
            if errs:
                raise ValueError("schema: " + "; ".join(errs[:5]))
        return value, None

    # ---- single item -----------------------------------------------------
    def run(self, job_name: str, item, index: int = 0, stream: bool = False,
            on_delta=None, use_cache: bool | None = None, model: str | None = None) -> Result:
        job = self.cfg.jobs.get(job_name)
        if job is None:
            raise JobError(f"no job '{job_name}' in {self.cfg.path}")

        key = item.get("key") if isinstance(item, dict) else None
        msgs = self.messages(job, item)
        spec = [model] if model else job.model
        try:
            chain = self.registry.resolve(spec)
        except ValueError as e:
            return Result(job=job_name, index=index, key=key, ok=False, error=str(e))

        cache_on = job.cache if use_cache is None else use_cache
        # response_format shape varies per provider, so the key records the modes
        modes = [self.registry.providers[t.provider].schema_mode for t in chain]
        payload = {"job": job_name, "model": [str(t) for t in chain], "messages": msgs,
                   "params": job.params, "format": job.format, "schema": job.schema,
                   "modes": modes}
        ck = self.cache.key(payload)
        if cache_on:
            hit = self.cache.get(ck)
            if hit:
                r = Result(job=job_name, index=index, key=key, ok=True, cached=True,
                           text=hit.get("text", ""), parsed=hit.get("parsed"),
                           target=hit.get("model"), provider=(hit.get("model") or "").split("/")[0] or None,
                           model="/".join((hit.get("model") or "").split("/")[1:]) or None,
                           usage=hit.get("usage") or {})
                return r

        tried: list[str] = []
        last_err = None
        for target in chain:
            prov = self.registry.providers[target.provider]
            tried.append(str(target))
            hint = self._schema_hint(job, prov)
            sent = msgs + [{"role": "system", "content": hint}] if hint else msgs
            try:
                chat: ChatResult = prov.chat(
                    sent, target.model, self._params(job, prov),
                    stream=stream and on_delta is not None, on_delta=on_delta)
            except ProviderError as e:
                last_err = str(e)
                continue
            try:
                parsed, _ = self._parse(job, chat.text)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = f"{target}: {e}"
                continue
            res = Result(
                job=job_name, index=index, key=key, ok=True, text=chat.text, parsed=parsed,
                target=str(target), provider=target.provider, model=target.model,
                attempts=chat.attempts, latency=chat.latency, usage=chat.usage, tried=tried)
            if cache_on:
                self.cache.put(ck, {"text": chat.text, "parsed": parsed,
                                    "model": str(target), "usage": chat.usage})
            return res

        notes = f" (skipped: {'; '.join(chain.notes)})" if chain.notes else ""
        return Result(job=job_name, index=index, key=key, ok=False,
                      target=tried[-1] if tried else None, tried=tried,
                      error=(last_err or "no model available") + notes)

    # ---- fan-out ---------------------------------------------------------
    def run_job(self, job_name: str, items, stream: bool = False,
                on_delta=None, use_cache: bool | None = None,
                on_result=None, refresh: bool = True, model: str | None = None) -> list[Result]:
        job = self.cfg.jobs.get(job_name)
        if job is None:
            raise JobError(f"no job '{job_name}' in {self.cfg.path}")
        items = list(items)
        if refresh:
            self.registry.refresh()
        if stream and len(items) > 1:
            stream = False                      # deltas from parallel calls interleave

        results: list[Result | None] = [None] * len(items)

        def work(pair):
            i, item = pair
            res = self.run(job_name, item, index=i, stream=stream,
                           on_delta=on_delta, use_cache=use_cache, model=model)
            if on_result:
                on_result(res)
            return i, res

        workers = max(1, min(job.max_workers, len(items) or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in pool.map(work, list(enumerate(items))):
                results[i] = res
        return [r for r in results if r is not None]

    # ---- multi-stage -----------------------------------------------------
    def _stage_input(self, res: Result, original, selector: str) -> str:
        """Turn a stage result into the prompt text for the next stage."""
        if selector == "text":
            return res.text
        if selector == "parsed":
            return res.parsed if isinstance(res.parsed, str) else json.dumps(res.parsed, ensure_ascii=False)
        if selector == "json":
            value = res.parsed if res.parsed is not None else res.text
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if selector.startswith("field:"):
            field = selector[6:]
            if isinstance(original, dict) and field in original:
                value = original[field]
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            return res.text
        raise JobError(f"unknown stage input selector: {selector!r}")

    def run_pipeline(self, name: str, items, use_cache: bool | None = None,
                     on_stage=None, refresh: bool = True) -> list[dict]:
        """Chain jobs: each stage feeds the next, per item, with per-stage models."""
        pipe = self.cfg.pipelines.get(name)
        if pipe is None:
            raise JobError(f"no pipeline '{name}' in {self.cfg.path}")
        items = list(items)
        if refresh:
            self.registry.refresh()

        # Seed each item with a pseudo-result so stage 1 has a predecessor.
        state = [{"item": it, "stages": [Result(job="__input__", index=i,
                                                key=it.get("key") if isinstance(it, dict) else None,
                                                ok=True, text=_item_text(it),
                                                parsed=it)]}
                 for i, it in enumerate(items)]
        for n, stage in enumerate(pipe.stages, 1):
            feed = []
            for st in state:
                prev = st["stages"][-1]
                text = self._stage_input(prev, st["item"], stage.input)
                nxt = {"key": prev.key, "input": text}
                if stage.keep and isinstance(st["item"], dict):
                    for k, v in st["item"].items():
                        nxt.setdefault(k, v)
                feed.append(nxt)
            results = self.run_job(stage.job, feed, use_cache=use_cache, refresh=False,
                                   model=stage.model)
            for st, res in zip(state, results):
                st["stages"].append(res)
            if on_stage:
                on_stage(n, stage, results)
            if any(not r.ok for r in results) and pipe.fail_fast:
                break

        out = []
        for st in state:
            ran = st["stages"][1:]              # [0] is the input seed
            final = ran[-1] if ran else None
            out.append({
                "pipeline": name,
                "key": st["item"].get("key") if isinstance(st["item"], dict) else None,
                "ok": bool(final and final.ok),
                "input": st["item"],
                "text": final.text if final else "",
                "parsed": final.parsed if final else None,
                "model": final.target if final else None,
                "error": final.error if final else "no stages ran",
                "stages": [r.record() for r in ran],
            })
        return out

    def summarize(self, results: list[Result]) -> dict:
        ok = sum(1 for r in results if r.ok)
        return {
            "n": len(results), "ok": ok, "failed": len(results) - ok,
            "cached": sum(1 for r in results if r.cached),
            "seconds": round(sum(r.latency for r in results), 2),
            "tokens": sum((r.usage.get("total_tokens") or 0) for r in results),
            "by_model": _counts(r.target for r in results),
        }


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return out


def _item_text(item) -> str:
    """Prompt text for an input item, mirroring Harness.messages' rules."""
    if isinstance(item, dict):
        body = item.get("input", item.get("text", json.dumps(item, ensure_ascii=False)))
        return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return str(item)


def load_items(path: str) -> list:
    """Read job input: JSON array, JSONL, or one line per item. '-' is stdin."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text("utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    lines = [line for line in text.splitlines() if line.strip()]
    if path == "-":
        # No suffix to trust: treat all-JSON lines as JSONL, else plain lines.
        try:
            return [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            return lines
    suffix = Path(path).suffix
    if suffix in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in lines]
    if suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return lines


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
