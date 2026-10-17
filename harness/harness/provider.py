"""OpenAI-compatible HTTP client on stdlib urllib: discovery, chat, SSE streaming, retries."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ProviderError(Exception):
    """A provider call failed. `retryable` marks transient transport/status faults."""

    def __init__(self, provider: str, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(f"[{provider}] {message}" + (f" (HTTP {status})" if status else ""))
        self.provider = provider
        self.status = status
        self.retryable = retryable


@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    attempts: int = 1
    raw: dict = field(default_factory=dict)


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str | None = None
    timeout: float = 180.0
    max_retries: int = 2
    strict: bool = False
    default: bool = False
    schema_mode: str = "json_object"   # json_object | json_schema | prompt
    headers: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    listed: list[str] = field(default_factory=list)
    static: list[str] = field(default_factory=list)   # config `models`, always kept
    discovered_at: float = 0.0

    def url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _open(self, path: str, payload: dict, stream: bool):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url(path), data=body, method="POST")
        req.add_header("content-type", "application/json")
        req.add_header("accept", "text/event-stream" if stream else "application/json")
        if self.api_key:
            req.add_header("authorization", f"Bearer {self.api_key}")
        for k, v in self.headers.items():
            req.add_header(k, v)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _get(self, path: str):
        req = urllib.request.Request(self.url(path), method="GET")
        req.add_header("accept", "application/json")
        if self.api_key:
            req.add_header("authorization", f"Bearer {self.api_key}")
        for k, v in self.headers.items():
            req.add_header(k, v)
        return urllib.request.urlopen(req, timeout=min(self.timeout, 30.0))

    def list_models(self) -> list[str]:
        """Fetch served model ids from /models; caches them on the provider."""
        try:
            with self._get("models") as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            raise ProviderError(self.name, f"model discovery failed: {_blurb(e)}", e.code) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(self.name, f"unreachable: {e}", retryable=True) from e
        ids = []
        for item in (data.get("data") or []):
            mid = item.get("id") or item.get("model") or item.get("name")
            if mid:
                ids.append(str(mid))
        # Config-declared ids stay listed even when the endpoint omits them.
        self.listed = ids + [m for m in self.static if m not in ids]
        self.discovered_at = time.time()
        return self.listed

    def chat(self, messages: list[dict], model: str, params: dict | None = None,
             stream: bool = False, on_delta=None) -> ChatResult:
        """POST /chat/completions with backoff on transient faults. Raises ProviderError."""
        params = params or {}
        payload = {"model": model, "messages": messages, "stream": stream}
        payload.update(self.extra)                 # provider defaults first
        for key in ("temperature", "top_p", "max_tokens", "seed", "stop",
                    "presence_penalty", "frequency_penalty", "response_format", "user"):
            if params.get(key) is not None:
                payload[key] = params[key]         # per-job params win
        payload.update(params.get("extra") or {})

        last: ProviderError | None = None
        for attempt in range(1, self.max_retries + 2):
            started = time.time()
            try:
                with self._open("chat/completions", payload, stream) as resp:
                    text, finish, usage, echoed, raw = (
                        _drain_sse(resp, on_delta) if stream else _read_json(resp))
                return ChatResult(
                    text=text, provider=self.name, model=echoed or model,
                    finish_reason=finish, usage=usage, latency=time.time() - started,
                    attempts=attempt, raw=raw)
            except urllib.error.HTTPError as e:
                err = ProviderError(self.name, _blurb(e), e.code, e.code in RETRY_STATUS)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = ProviderError(self.name, f"transport: {e}", retryable=True)
            last = err
            if not err.retryable or attempt > self.max_retries:
                raise err
            time.sleep(min(8.0, 0.6 * 2 ** (attempt - 1)))
        raise last  # unreachable


def _blurb(e: urllib.error.HTTPError) -> str:
    try:
        detail = e.read().decode("utf-8", "replace").strip()
    except Exception:
        detail = ""
    if detail:
        return (detail.splitlines()[0])[:300]
    return e.reason or str(e)


def _read_json(resp):
    data = json.loads(resp.read().decode("utf-8", "replace"))
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    text = msg.get("content")
    if text is None:
        text = choice.get("text") or ""
    return text, choice.get("finish_reason"), data.get("usage") or {}, data.get("model"), data


def _drain_sse(resp, on_delta):
    """Consume `data: {...}` events, forwarding content deltas; returns the assembled reply."""
    parts, finish, usage, echoed = [], None, {}, None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        echoed = data.get("model") or echoed
        if data.get("usage"):
            usage = data["usage"]
        for choice in (data.get("choices") or []):
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece is None:
                piece = (choice.get("message") or {}).get("content")
            if piece:
                parts.append(piece)
                if on_delta:
                    on_delta(piece)
            finish = choice.get("finish_reason") or finish
    return "".join(parts), finish, usage, echoed, {}
