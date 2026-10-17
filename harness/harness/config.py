"""Config loading: providers, aliases, jobs. TOML via stdlib tomllib, ${ENV} expansion."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")

SEARCH = (
    "config.toml",
    ".harness/config.toml",
    "harness.toml",
    ".config/harness/config.toml",
)

PARAM_KEYS = ("temperature", "top_p", "max_tokens", "seed", "stop",
              "presence_penalty", "frequency_penalty")


def expand(value, env=None):
    """Recursively expand ${VAR} and ${VAR:-default} in strings inside a config value."""
    env = os.environ if env is None else env
    if isinstance(value, str):
        def sub(m):
            return env.get(m.group(1)) or (m.group(2) if m.group(2) is not None else "")
        return ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    return value


@dataclass
class ProviderCfg:
    name: str
    base_url: str
    api_key: str | None = None
    timeout: float = 180.0
    max_retries: int = 2
    strict: bool = False          # reject models absent from /v1/models
    default: bool = False         # preferred target for bare model ids
    models: list[str] = field(default_factory=list)   # static hints, merged with discovery
    headers: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)         # merged into every request body


@dataclass
class JobCfg:
    name: str
    model: list[str] = field(default_factory=list)    # ordered fallback chain
    system: str = ""
    template: str = ""            # {input}/{key} prompt template; empty = item passed as-is
    format: str = "text"          # text | json | schema
    schema: dict | None = None
    params: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    max_workers: int = 4
    cache: bool = True


@dataclass
class StageCfg:
    job: str
    input: str = "text"           # text | parsed | json | field:<name>
    model: str | None = None      # per-stage model override
    keep: bool = True             # carry the original item alongside the result


@dataclass
class PipelineCfg:
    name: str
    stages: list[StageCfg] = field(default_factory=list)
    fail_fast: bool = False       # stop the pipeline when a stage item fails


@dataclass
class Config:
    path: str
    providers: dict[str, ProviderCfg]
    aliases: dict[str, list[str]]
    jobs: dict[str, JobCfg]
    pipelines: dict[str, PipelineCfg]
    cache_dir: str
    raw: dict = field(default_factory=dict)

    @property
    def default_provider(self) -> str:
        for p in self.providers.values():
            if p.default:
                return p.name
        return next(iter(self.providers))


def _provider(name: str, tbl: dict) -> ProviderCfg:
    key = tbl.get("api_key")
    if not key and tbl.get("api_key_env"):
        key = os.environ.get(str(tbl["api_key_env"]))
    return ProviderCfg(
        name=name,
        base_url=str(tbl["base_url"]).rstrip("/"),
        api_key=key or None,
        timeout=float(tbl.get("timeout", 180.0)),
        max_retries=int(tbl.get("max_retries", 2)),
        strict=bool(tbl.get("strict", False)),
        default=bool(tbl.get("default", False)),
        models=[str(m) for m in tbl.get("models", [])],
        headers={str(k): str(v) for k, v in (tbl.get("headers") or {}).items()},
        extra=dict(tbl.get("extra") or {}),
    )


def _job(name: str, tbl: dict) -> JobCfg:
    spec = tbl.get("model", [])
    if isinstance(spec, str):
        spec = [spec]
    params = {k: tbl[k] for k in PARAM_KEYS if k in tbl}
    params.update(tbl.get("params") or {})
    return JobCfg(
        name=name,
        model=[str(s) for s in spec],
        system=str(tbl.get("system", "")),
        template=str(tbl.get("template", "")),
        format=str(tbl.get("format", "text")),
        schema=tbl.get("schema"),
        params=params,
        extra=dict(tbl.get("extra") or {}),
        max_workers=int(tbl.get("max_workers", 4)),
        cache=bool(tbl.get("cache", True)),
    )


def _pipeline(name: str, tbl: dict) -> PipelineCfg:
    stages = []
    for raw in (tbl.get("stages") or []):
        if isinstance(raw, str):
            raw = {"job": raw}
        if not raw.get("job"):
            raise SystemExit(f"pipeline '{name}': stage needs a job name")
        stages.append(StageCfg(
            job=str(raw["job"]),
            input=str(raw.get("input", "text")),
            model=(str(raw["model"]) if raw.get("model") else None),
            keep=bool(raw.get("keep", True))))
    if not stages:
        raise SystemExit(f"pipeline '{name}' has no stages")
    return PipelineCfg(name=name, stages=stages, fail_fast=bool(tbl.get("fail_fast", False)))


def load(path: str | os.PathLike | None = None) -> Config:
    """Load config from an explicit path, $HARNESS_CONFIG, or the search list."""
    chosen = None
    if path:
        chosen = Path(path).expanduser()
        if not chosen.exists():
            raise SystemExit(f"config not found: {chosen}")
    if chosen is None:
        env = os.environ.get("HARNESS_CONFIG")
        if env:
            chosen = Path(env).expanduser()
            if not chosen.exists():
                raise SystemExit(f"HARNESS_CONFIG points nowhere: {chosen}")
    if chosen is None:
        for cand in SEARCH:
            if Path(cand).exists():
                chosen = Path(cand)
                break
    if chosen is None:
        raise SystemExit("no config found; create ./config.toml (see README)")

    data = expand(tomllib.loads(chosen.read_text("utf-8")))

    providers = {}
    for name, tbl in (data.get("providers") or {}).items():
        providers[name] = _provider(name, tbl)
    for tbl in data.get("provider") or []:                      # [[provider]] array form
        name = str(tbl["name"])
        providers[name] = _provider(name, tbl)
    if not providers:
        raise SystemExit(f"{chosen}: no [providers.*] entries")

    aliases = {}
    for alias, target in (data.get("aliases") or {}).items():
        aliases[str(alias)] = [str(t) for t in ([target] if isinstance(target, str) else target)]

    jobs = {}
    for name, tbl in (data.get("jobs") or {}).items():
        jobs[name] = _job(name, tbl)
    for tbl in data.get("job") or []:
        jobs[str(tbl["name"])] = _job(str(tbl["name"]), tbl)

    pipelines = {}
    for name, tbl in (data.get("pipelines") or {}).items():
        pipelines[str(name)] = _pipeline(str(name), tbl)
    for tbl in data.get("pipeline") or []:
        pipelines[str(tbl["name"])] = _pipeline(str(tbl["name"]), tbl)
    for name, pipe in pipelines.items():
        missing = [s.job for s in pipe.stages if s.job not in jobs]
        if missing:
            raise SystemExit(f"pipeline '{name}' references unknown job(s): {', '.join(missing)}")

    return Config(
        path=str(chosen),
        providers=providers,
        aliases=aliases,
        jobs=jobs,
        pipelines=pipelines,
        cache_dir=str(data.get("cache_dir", ".harness/cache")),
        raw=data,
    )
