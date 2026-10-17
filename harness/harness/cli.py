"""CLI: models, jobs, ask, run, ping. Exit 0 all-ok, 1 any failure, 2 usage/config error."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

from . import __version__
from .config import JobCfg, load
from .pipeline import Harness, JobError, Result, load_items, stamp


def _harness(args) -> Harness:
    return Harness(load(args.config))


def _chain(h: Harness, job_name: str, override: str | None) -> str:
    job = h.cfg.jobs.get(job_name)
    spec = [override] if override else (job.model if job else [])
    try:
        chain = h.registry.resolve(spec)
    except ValueError as e:
        return f"!! {e}"
    return str(chain) + (f"  !! skipped: {'; '.join(chain.notes)}" if chain.notes else "")


def _emit(res: Result, args) -> None:
    if args.json:
        sys.stdout.write(json.dumps(res.record(), ensure_ascii=False) + "\n")
        return
    tag = "cache" if res.cached else (res.target or "-")
    if res.ok:
        body = res.text if args.verbose else " ".join(res.text.split())
        head = (f"ok   #{res.index} [{tag}] {res.latency:.1f}s"
                + (f" key={res.key}" if res.key else ""))
        block = head + "\n     " + "\n     ".join(body.splitlines() or [""])
    else:
        block = f"FAIL #{res.index} tried={', '.join(res.tried) or '-'}: {res.error}"
    sys.stdout.write(block + "\n")     # one write keeps parallel output readable
    sys.stdout.flush()


# ---- commands ------------------------------------------------------------

def cmd_models(args) -> int:
    h = _harness(args)
    h.registry.refresh(ttl=0)
    errors = h.registry.discovery_errors()
    if args.json:
        print(json.dumps({
            "providers": {n: p.listed for n, p in h.registry.providers.items()},
            "errors": errors,
        }, indent=2))
        return 0 if not errors else 1
    width = max(len(n) for n in h.registry.providers)
    for name, prov in h.registry.providers.items():
        state = errors.get(name)
        flags = "".join(f" {k}" for k, on in (("default", prov.default), ("strict", prov.strict)) if on)
        print(f"{name:<{width}}  {prov.base_url}{flags}")
        if state:
            print(f"{'':<{width}}  !! {state}")
        for mid in prov.listed:
            print(f"{'':<{width}}    {mid}")
        if not prov.listed and not state:
            print(f"{'':<{width}}    (no models served yet)")
    print(f"\n{sum(len(p.listed) for p in h.registry.providers.values())} models "
          f"across {len(h.registry.providers)} providers"
          + (f", {len(errors)} unreachable" if errors else ""))
    return 0 if not errors else 1


def cmd_jobs(args) -> int:
    h = _harness(args)
    h.registry.refresh()
    if not h.cfg.jobs:
        print("no jobs configured")
        return 0
    width = max(len(j) for j in h.cfg.jobs)
    for name, job in h.cfg.jobs.items():
        print(f"{name:<{width}}  fmt={job.format:<5} workers={job.max_workers} "
              f"cache={int(job.cache)}")
        print(f"{'':<{width}}  chain: {_chain(h, name, args.model)}")
        if job.system:
            print(f"{'':<{width}}  system: {' '.join(job.system.split())[:110]}")
    return 0


def cmd_ping(args) -> int:
    h = _harness(args)
    h.registry.refresh(ttl=0)
    errors = h.registry.discovery_errors()
    width = max(len(n) for n in h.registry.providers)
    for name, prov in h.registry.providers.items():
        if name in errors:
            print(f"FAIL {name:<{width}} {prov.base_url}: {errors[name]}")
        else:
            print(f"ok   {name:<{width}} {prov.base_url} -> {len(prov.listed)} models")
    return 1 if errors else 0


def cmd_ask(args) -> int:
    h = _harness(args)
    h.registry.refresh()
    job = args.job
    if job is None:
        if not args.model:
            print("ask needs -j <job> or -m <model>", file=sys.stderr)
            return 2
        job = "__adhoc__"
        h.cfg.jobs[job] = JobCfg(
            name=job, model=[""], system=args.system or "", format=args.format,
            params={"temperature": args.temperature} if args.temperature is not None else {},
            cache=False)
    elif job not in h.cfg.jobs:
        print(f"no job '{job}'; configured: {', '.join(h.cfg.jobs) or 'none'}", file=sys.stderr)
        return 2
    if args.model:
        h.cfg.jobs[job].model = [m.strip() for m in args.model.split(",")]
    if args.system:
        h.cfg.jobs[job].system = args.system
    if args.format != "text":
        h.cfg.jobs[job].format = args.format
    if args.stream:
        sys.stdout.write(f"[{_chain(h, job, args.model)}]\n")
    res = h.run(job, args.prompt, stream=args.stream,
                on_delta=(lambda d: sys.stdout.write(d)) if args.stream else None,
                use_cache=False if args.no_cache else None)
    if args.stream and res.ok:
        sys.stdout.write(f"\n[{res.target} {res.latency:.1f}s]\n")
        return 0
    _emit(res, args)
    return 0 if res.ok else 1


def cmd_run(args) -> int:
    h = _harness(args)
    if args.job not in h.cfg.jobs:
        print(f"no job '{args.job}'; configured: {', '.join(h.cfg.jobs) or 'none'}", file=sys.stderr)
        return 2
    if args.model:
        h.cfg.jobs[args.job].model = [m.strip() for m in args.model.split(",")]
    if args.workers:
        h.cfg.jobs[args.job].max_workers = args.workers
    items = load_items(args.infile) if args.infile else list(args.prompt or [])
    if not items:
        print("no input items", file=sys.stderr)
        return 2

    out = None
    path = args.out or f".harness/out/{args.job}-{stamp()}.jsonl"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = open(path, "w", encoding="utf-8")
    except OSError as e:
        print(f"cannot open {path}: {e}", file=sys.stderr)
        return 2

    sink_lock = threading.Lock()

    def sink(res: Result) -> None:
        with sink_lock:                     # workers finish concurrently
            out.write(json.dumps(res.record(), ensure_ascii=False) + "\n")
            out.flush()
            if not args.quiet:
                _emit(res, args)

    results = h.run_job(args.job, items, stream=args.stream,
                        on_delta=(lambda d: sys.stdout.write(d)) if args.stream and len(items) == 1 else None,
                        use_cache=False if args.no_cache else None, on_result=sink)
    out.close()
    stats = h.summarize(results)
    print(f"\n{stats['ok']}/{stats['n']} ok  cached={stats['cached']} "
          f"failed={stats['failed']} {stats['seconds']}s tokens={stats['tokens']} -> {path}")
    for model, n in sorted(stats["by_model"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {model}")
    return 0 if stats["failed"] == 0 else 1


def cmd_pipelines(args) -> int:
    h = _harness(args)
    h.registry.refresh()
    if not h.cfg.pipelines:
        print("no pipelines configured")
        return 0
    for name, pipe in h.cfg.pipelines.items():
        hops = " -> ".join(
            f"{s.job}" + (f"@{s.model}" if s.model else "") + (f"[{s.input}]" if s.input != "text" else "")
            for s in pipe.stages)
        print(f"{name}  fail_fast={int(pipe.fail_fast)}")
        print(f"  {hops}")
        for s in pipe.stages:
            print(f"    {s.job:<14} -> {_chain(h, s.job, s.model)}")
    return 0


def cmd_pipe(args) -> int:
    h = _harness(args)
    if args.pipeline not in h.cfg.pipelines:
        print(f"no pipeline '{args.pipeline}'; configured: "
              f"{', '.join(h.cfg.pipelines) or 'none'}", file=sys.stderr)
        return 2
    items = load_items(args.infile) if args.infile else list(args.prompt or [])
    if not items:
        print("no input items", file=sys.stderr)
        return 2

    def on_stage(n, stage, results):
        if args.quiet:
            return
        ok = sum(1 for r in results if r.ok)
        models = ", ".join(sorted({str(r.target) for r in results if r.target}))
        print(f"stage {n} {stage.job} -> {ok}/{len(results)} ok [{models}]")

    rows = h.run_pipeline(args.pipeline, items,
                          use_cache=False if args.no_cache else None, on_stage=on_stage)

    path = args.out or f".harness/out/{args.pipeline}-{stamp()}.jsonl"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"cannot open {path}: {e}", file=sys.stderr)
        return 2

    if args.json:
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif not args.quiet:
        for row in rows:
            if row["ok"]:
                body = row["text"] if args.verbose else " ".join(str(row["text"]).split())
                sys.stdout.write(f"ok   [{row['model']}] {body}\n")
            else:
                sys.stdout.write(f"FAIL {row['key'] or row['input']}: {row['error']}\n")
        sys.stdout.flush()

    failed = sum(1 for r in rows if not r["ok"])
    print(f"\n{len(rows) - failed}/{len(rows)} ok -> {path}")
    return 0 if failed == 0 else 1


# ---- parser --------------------------------------------------------------

def build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description=__doc__)
    p.add_argument("--version", action="version", version=f"harness {__version__}")
    p.add_argument("-c", "--config", help="config.toml path")
    p.add_argument("-m", "--model", help="override model spec: alias, provider/id, id, or a,b fallback chain")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-v", "--verbose", action="store_true", help="keep raw model formatting")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress per-item lines")

    # Accept the shared flags on either side of the subcommand; SUPPRESS keeps
    # the pre-subcommand value when the subcommand-side flag is absent.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=argparse.SUPPRESS)
    common.add_argument("-m", "--model", default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", aliases=["ls"], parents=[common], help="list providers and served models")
    m.set_defaults(func=cmd_models)

    j = sub.add_parser("jobs", parents=[common], help="show jobs and their resolved model chains")
    j.set_defaults(func=cmd_jobs)

    g = sub.add_parser("ping", parents=[common], help="check provider reachability")
    g.set_defaults(func=cmd_ping)

    a = sub.add_parser("ask", parents=[common], help="one prompt")
    a.add_argument("prompt")
    a.add_argument("-j", "--job", help="job profile to use")
    a.add_argument("-s", "--system")
    a.add_argument("-f", "--format", choices=["text", "json", "schema"], default="text")
    a.add_argument("-t", "--temperature", type=float)
    a.add_argument("--stream", action="store_true")
    a.add_argument("--no-cache", action="store_true")
    a.set_defaults(func=cmd_ask)

    r = sub.add_parser("run", parents=[common], help="batch a job over items")
    r.add_argument("job")
    r.add_argument("-i", "--in", dest="infile", help="items: .json array, .jsonl, or lines")
    r.add_argument("-p", "--prompt", action="append", help="inline item (repeatable)")
    r.add_argument("-o", "--out", help="output jsonl (default .harness/out/<job>-<stamp>.jsonl)")
    r.add_argument("-w", "--workers", type=int)
    r.add_argument("--stream", action="store_true")
    r.add_argument("--no-cache", action="store_true")
    r.set_defaults(func=cmd_run)

    pl = sub.add_parser("pipelines", aliases=["pipes"], parents=[common],
                        help="show configured pipelines and their model chains")
    pl.set_defaults(func=cmd_pipelines)

    pp = sub.add_parser("pipe", parents=[common], help="run a multi-stage pipeline")
    pp.add_argument("pipeline")
    pp.add_argument("-i", "--in", dest="infile", help="items: .json array, .jsonl, or lines")
    pp.add_argument("-p", "--prompt", action="append", help="inline item (repeatable)")
    pp.add_argument("-o", "--out", help="output jsonl (default .harness/out/<pipeline>-<stamp>.jsonl)")
    pp.add_argument("--no-cache", action="store_true")
    pp.set_defaults(func=cmd_pipe)
    return p


def main(argv=None) -> int:
    args = build().parse_args(argv)
    try:
        return args.func(args)
    except JobError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
