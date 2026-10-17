"""Model registry: live discovery, aliases, and spec -> (provider, model) resolution."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Config
from .provider import Provider, ProviderError

DISCOVERY_TTL = 30.0


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    listed: bool = True

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}" + ("" if self.listed else " (unlisted)")


class Chain(list):
    """Resolved fallback chain plus notes about specs that were skipped."""

    def __init__(self, items=()):
        super().__init__(items)
        self.notes: list[str] = []

    def __str__(self) -> str:
        return " -> ".join(str(t) for t in self)


class Registry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers: dict[str, Provider] = {}
        for name, p in cfg.providers.items():
            self.providers[name] = Provider(
                name=name, base_url=p.base_url, api_key=p.api_key, timeout=p.timeout,
                max_retries=p.max_retries, strict=p.strict, default=p.default,
                headers=p.headers, extra=p.extra, listed=list(p.models))
        self.aliases = cfg.aliases
        self.errors: dict[str, str] = {}

    # ---- discovery -------------------------------------------------------
    def refresh(self, provider: str | None = None, ttl: float = DISCOVERY_TTL) -> dict[str, list[str]]:
        """Refresh cached model lists, skipping providers discovered within `ttl`."""
        out = {}
        for name, prov in self.providers.items():
            if provider and name != provider:
                continue
            if ttl and prov.discovered_at and time.time() - prov.discovered_at < ttl:
                out[name] = prov.listed
                continue
            try:
                out[name] = prov.list_models()
                self.errors.pop(name, None)
            except ProviderError as e:
                # Keep static hints, remember the fault, and allow a fast retry.
                out[name] = prov.listed
                prov.discovered_at = 0.0
                self.errors[name] = str(e)
        return out

    def discovery_errors(self) -> dict[str, str]:
        return dict(self.errors)

    def catalog(self) -> list[tuple[str, str, bool]]:
        """(provider, model, from_discovery) for every known model."""
        rows = []
        for name, prov in self.providers.items():
            for mid in prov.listed:
                rows.append((name, mid, mid not in self.cfg.providers[name].models))
        return rows

    def known(self, provider: str, model: str) -> bool:
        return model in self.providers[provider].listed

    # ---- resolution ------------------------------------------------------
    def resolve(self, spec: str | list[str]) -> "Chain":
        """Expand a spec (alias, provider/model, bare id, fuzzy) into a fallback chain.

        A spec that cannot resolve (e.g. a strict provider that does not serve it)
        is recorded in `notes` and skipped, so the rest of a fallback chain still runs.
        """
        specs = [spec] if isinstance(spec, str) else list(spec)
        chain = Chain()
        for raw in specs:
            raw = raw.strip()
            if not raw:
                continue
            try:
                for item in self._expand(raw):
                    if item not in chain:
                        chain.append(item)
            except ValueError as e:
                chain.notes.append(str(e))
        if not chain:
            raise ValueError("; ".join(chain.notes) or f"model spec resolved to nothing: {spec!r}")
        return chain

    def _expand(self, raw: str) -> list[Target]:
        if raw in self.aliases:
            expanded = []
            for alias_target in self.aliases[raw]:
                expanded.extend(self._expand(alias_target))
            return expanded
        if not raw:
            return []

        head, _, rest = raw.partition("/")
        if rest and head in self.providers:                       # provider-qualified
            prov = self.providers[head]
            if prov.strict and rest not in prov.listed:
                raise ValueError(f"{head} is strict and does not serve {rest!r}; loaded: "
                                 f"{', '.join(prov.listed) or 'none'}")
            return [Target(head, rest, rest in prov.listed)]

        hits = [Target(name, raw, True) for name, prov in self.providers.items() if raw in prov.listed]
        if hits:
            return hits

        fuzzy = [Target(name, mid, True)
                 for name, prov in self.providers.items()
                 for mid in prov.listed
                 if mid.lower().startswith(raw.lower()) or raw.lower() in mid.lower()]
        if fuzzy:
            return fuzzy

        # A bare id lands on the default provider; strictness applies there.
        default = self.cfg.default_provider
        if self.providers[default].strict:
            raise ValueError(f"{default} is strict and does not serve {raw!r}; loaded: "
                             f"{', '.join(self.providers[default].listed) or 'none'}")
        # Models load lazily: an unlisted id is a real attempt, flagged for the operator.
        return [Target(default, raw, listed=False)]
