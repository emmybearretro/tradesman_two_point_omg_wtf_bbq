# HARNESS HANDOFF — everything, no context needed

Read this before touching `/home/hc/hackathon/harness`. Companion doc: `handoff-training.md`
(the blueprint model training side, separate repo area).

## 1. What the harness is for

Offline-first lumber-quote pipeline. Two ingest legs converge, then deterministic math:

```
audio  -> STT        -> transcript -> intake (schema) -> context
image  -> vision LLM -> blueprint  (schema) -> context
                         both accepted? -> lumber calc + pricing -> quote dict -> PDF
```

Back on internet: phone leg (TTS out, transcript in) negotiates with a human, pricing is
updated, final quote goes out for approval. **Deferred — needs connectivity.** Design the
seam, do not build it.

Non-negotiable: the money path (lumber calc, pricing, PDF) is **deterministic code, never an
LLM**. The LLM only extracts numbers from media. A model must never do arithmetic that ends
up on a customer's invoice.

## 2. Current state (2026-10-17)

- **Location:** `/home/hc/hackathon/harness`. Git root is the **parent** `/home/hc/hackathon`
  — `git status` from inside `harness/` prefixes paths with `harness/`. Stray untracked files
  at the parent root (`camera_upload_server.py`, `fix_remote_code.py`, `static/`) are **not
  ours**; leave them alone.
- **Test suite: 75 tests, green.** `cd /home/hc/hackathon/harness && python3 tests/run_tests.py`
  (~2.9s). Runs an in-process OpenAI-compatible stub on an ephemeral port — no external
  service needed.
- **Nothing is committed.** Uncommitted diff: `harness/pipeline.py` (+8/-2),
  `harness/tests/run_tests.py` (+40/-1). Untracked: `harness/calc.py`. Commit before
  continuing.
- **Zero third-party dependencies.** Python 3.11+ stdlib only (`tomllib`, `urllib`).
  Python 3.15.0 via mise. Keep it that way — pip installs are forbidden for the harness.
- Ollama is up on `127.0.0.1:11434` with **zero models loaded**. The old mock provider on
  `127.0.0.1:8080` is dead; the suite does not need it.

## 3. What is proven working (text-only chat harness)

Read `README.md` — it is accurate except for the three drift items in §7.

- `harness/cli.py` (332 lines): `models`/`ls`, `jobs`, `ping`, `ask`, `run`,
  `pipelines`/`pipes`, `pipe`. Entry points `./harness.py` and `python3 -m harness`.
  Exit codes: 0 all ok, 1 any item failed, 2 usage/config.
- `harness/provider.py` (176): OpenAI-compatible client on stdlib `urllib`. Discovery
  (`GET /v1/models`), SSE streaming, backoff on 429/5xx/transport, optional
  `Authorization: Bearer` (header omitted entirely when no key).
- `harness/registry.py` (142): alias expansion (alias-of-alias, cycle detection),
  `provider/model` qualification, bare-id resolution (serving provider -> fuzzy
  prefix/substring -> default provider), `strict` providers refuse unlisted ids,
  unlisted ids on permissive providers are attempted and flagged `(unlisted)`.
  Static `models = [...]` hints survive discovery (union, not replace).
- `harness/pipeline.py` (448): jobs, fallback chains, JSON extraction from prose/``` fences,
  local schema validation (type/required/properties/items/enum/const/min/max), disk cache
  keyed on a sha256 of the request payload, `ThreadPoolExecutor` fan-out, multi-stage
  chaining with `input = text|parsed|json|field:<name>`, `keep`, `fail_fast`.
- `harness/config.py` (223): TOML, `${VAR}` / `${VAR:-default}` env expansion so keys never
  live in the file, search list (`./config.toml`, `./.harness/config.toml`, `./harness.toml`,
  `./.config/harness/config.toml`, `$HARNESS_CONFIG`, `-c`).
- Items: `-p` inline, `-i FILE` (`.json` array / `.jsonl` / plain lines), `-i -` stdin.
  `--json` records: `job index key ok model attempts latency cached usage error text parsed`.

## 4. Landed today, uncommitted: `schema_mode`

`format = "schema"` used to always send `response_format: json_schema` with `strict: true`.
Ollama's compat layer documents JSON mode, not `json_schema` — a strict-schema body is a 4xx
there, which the chain treats as a provider fault, so **every schema job would silently fall
through on the local endpoint**. Fixed by making the wire shape per-provider:

```toml
[providers.ollama]
schema_mode = "json_object"   # default: json_object | json_schema | prompt
```

- `json_object` (default) — sends `{"type":"json_object"}`, validates locally.
- `json_schema` — sends the full `json_schema` + `strict: true`. Opt-in, for OpenAI-class
  endpoints.
- `prompt` — sends **no** `response_format` at all, and appends a system message carrying
  `json.dumps(schema)`. For endpoints that reject structured-output fields outright.

Files: `ProviderCfg.schema_mode` + `SCHEMA_MODES` validation (`config.py`), `Provider.schema_mode`
(`provider.py`), wiring (`registry.py`), `Harness._params(job, prov)` /
`Harness._schema_hint(job, prov)` and per-target params in `run()` (`pipeline.py`). The cache
payload gained a `modes` list so a mode change invalidates cached replies.
Three new tests cover all three modes plus cache invalidation.

## 5. `harness/calc.py` — present, UNVERIFIED, needs your review

195 lines, untracked. A subagent wrote it before the user cancelled subagent work.
**No tests exist for it. It has never been run by the suite.**

Public API: `quote(context: dict, pricing: dict) -> dict`, `price_catalog(pricing: dict) -> dict`,
plus `BF_DIVISOR`, `SPEC_RE`. Pure functions, stdlib only.

Spot-checked by hand and it is arithmetically correct: 4x `2x6` @ 192in, SPF, #2, catalog
`2x6 = 0.95` -> `board_feet 64.0`, `material 60.80`, `waste_cost 6.08`, `labor 84.49`,
`delivery 85.00`, `subtotal 165.26`, `tax 13.63`, `min_charge 250.0` applied ->
`total 250.00` with a warning naming the labor delta. Return keys match the contract in §6.

Treat it as a draft: read it, then write `tests/test_calc.py` before wiring it in.

## 6. The design to build (agreed, not yet implemented)

Typed stages writing named context slots, with `require` as the fan-in gate:

```toml
[pipelines.quote]
stages = [
  { kind = "stt",    job = "transcribe", in = "audio",          as = "transcript" },
  { kind = "chat",   job = "intake",     in = "field:transcript", as = "intake" },
  { kind = "chat",   job = "blueprint",  in = "field:key", image = "field:images", as = "blueprint" },
  { kind = "calc",   job = "lumber",     pricing = "residential", as = "lumber",
    require = ["transcript", "blueprint"] },
  { kind = "render", job = "quote",      template = "quote", as = "quote", require = ["lumber"] },
]
```

- Item is a plain dict. Each stage writes `item[stage.as] = value`.
- `require`: slots must be present and truthy, else the stage and every later stage are
  skipped with `ok=False, error="missing: <slots>"`. This is "when both are accepted".
- Item input fields: `key` (label), `audio` (path), `images` (list of paths), `text`.
- Slots after a full run: `transcript` str, `intake` dict, `blueprint` dict, `lumber` dict,
  `quote` dict.
- `kind = "chat"` is today's behaviour and must stay the default so existing configs and all
  75 tests keep passing unchanged.

### Gap analysis vs today's code

| Leg | Nature | Status |
|---|---|---|
| audio -> transcript | STT, not chat completions | **missing** — needs multipart `POST /v1/audio/transcriptions` |
| image -> blueprint numbers | vision chat + schema | **missing** — `messages()` builds text-only `content` strings |
| numbers -> lumber + pricing | deterministic | `calc.py` draft, unwired, untested |
| quote -> PDF | deterministic | **missing** |
| context accumulation | merge into named slots | **partial** — `run_pipeline` chain-replaces via `_stage_input`, no slots, no `require` |
| phone + TTS | needs internet | deferred, seam only |

`Provider.chat()` passes `messages` straight through to the payload, so **multimodal content
parts need no provider change** — only `Harness.messages()` must be able to emit
`content: [{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}]`.
That is the smallest correct opening for the vision leg.

### Decisions already made by the user

- **STT:** NVIDIA model per stage where possible. Treat every stage as an OpenAI-compatible
  endpoint (NIM / Triton / vLLM / llama.cpp serving Nemotron, Parakeet/Canary ASR). Keep the
  harness model-agnostic; a stage is a base URL + model name.
- **Vision:** user is training the blueprint model (CAD + blueprint training, see
  `handoff-training.md`). It plugs in by name when ready. Do not block on it.
- **PDF:** Chromium headless HTML -> PDF. Verified on this box:
  `chromium --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer
  --print-to-pdf=out.pdf in.html` -> 15 KB, 1.3 s, text extractable by `pdftotext`.
- **Training data:** yes, export image + extracted-JSON pairs, **plus corrections**
  (`harness label` to edit a sample's numbers; corrected samples become the gold set, model
  output kept as baseline).

### Pricing config shape (agreed)

```toml
[pricing.residential]
currency = "USD"
waste_pct = 10
margin_pct = 20
delivery = 85.0
tax_rate = 0.0825
min_charge = 250.0
catalog = { "2x4" = 0.62, "2x6" = 0.95, "2x10" = 1.55, "1x6" = 0.78, "4x4" = 1.40 }  # $/board foot
species = { SPF = 1.0, DF = 1.25, SYP = 1.35, PT = 1.90 }
grades  = { "#2" = 1.0, "#1" = 1.18, "select" = 1.35 }
```

Board feet = `nominal_T * nominal_W * length_in / 144 * qty`.
`unit_price = catalog[spec] * species_mult * grade_mult`.
`material = sum(ext)`; `waste_cost = material * waste_pct/100`;
`labor = (material + waste_cost) * margin_pct/100`;
`subtotal = material + waste_cost + labor + delivery`; `tax = subtotal * tax_rate`;
`total = subtotal + tax`. If `total < min_charge`, raise `labor` by the gap and warn.
Unknown spec -> warning + member excluded. Unknown species/grade -> warning + multiplier 1.0.

## 7. Known drift to fix

- `README.md:158` says **72 tests**; there are **75**.
- `README.md:47-49` provider-key list omits `schema_mode`.
- `README.md:96-104` still claims `format = "schema"` always sends `json_schema` and adds a
  paragraph advising a `format = "json"` fallback target. Both are stale — `schema_mode`
  resolved that, and the fallback advice is now wrong.
- `config.toml` has no `schema_mode`, no `[pricing.*]`, no `[templates.*]` examples.

## 8. Box facts

- GPU **GTX 1650 Ti, 4 GB VRAM** (fp16 ~2 GB checkpoint ceiling; per-image load/unload, see
  `infer.py` in the parent repo). 16 cores, 62 GB RAM, 308 GB free on `/home`.
- Present: `ffmpeg`, `tesseract` (langs: afr, eng, osd), `magick`/`convert`, `chromium`,
  `pandoc`, `xelatex`, `pdftotext`, `qpdf`, `gs`, `numpy`, `PIL`, `sqlite3`.
- Absent: whisper (any flavour), `torch`, `reportlab`, `pypdf`, `fitz`, `pyaudio`,
  `sounddevice`, `twilio`, `deepgram`, `websockets`, `pytest`, `httpx`, `openai`, `sox`,
  `wkhtmltopdf`, `weasyprint`.
- Implication: PDF via Chromium (installed) not reportlab (not installed). STT needs an
  install or a subprocess adapter; 4 GB VRAM means CPU inference for ASR is the realistic
  local path.

## 9. Next steps, in order

1. Commit the uncommitted `schema_mode` work and the `calc.py` draft.
2. `tests/test_calc.py` for `calc.py` (§6 pricing rules are the spec). Fix what it finds.
3. `Harness.messages()` -> multimodal content parts (base64 `data:image/...` from
   `item["images"]`). Add a stub route that asserts the image part arrives.
4. `kind = "stt"`: multipart `POST /v1/audio/transcriptions` on `Provider`, ffmpeg
   normalize-to-16k-mono-wav as a pre-step. Pluggable `stt_cmd = "..."` escape hatch so the
   pipeline is testable before any engine is installed.
5. Typed stages + `as` + `require` in `run_pipeline` (fan-in gating, named slots).
6. `kind = "calc"` -> `harness/calc.py`; `kind = "render"` -> `harness/render.py` +
   `templates/quote.html` (`{{token}}` escaped, `{{&token}}` raw, stable `quote_id` from a
   sorted-key sha256, `$1,234.56` money format, Chromium invocation, temp-file cleanup on
   success and failure, `RenderError` carrying <=400 chars of chromium stderr).
7. Dataset export + `harness label` corrections (§6).
8. Phone-leg seam: interface + config keys + a documented stub. No implementation.

## 10. Gotchas

- **`edit` tool line anchors.** Two edits today clobbered code by anchoring on lines the tool
  had not displayed (a `JobCfg` dataclass body, then `[jobs.typed]`'s body in the test
  config). Rule: `read` the exact range immediately before every `edit`; never reuse a stale
  tag; never start/end a range mid-expression.
- **Test config is one shared TOML string** in `make_config()` (`tests/run_tests.py:115`).
  Adding a provider there changes resolution for every test (fuzzy matching, default
  provider, `models` exit codes). Add narrowly-scoped providers/jobs, and check which
  existing tests your addition perturbs.
- The stub keys scripted behaviour off the model id (`stub-400`, `stub-429-then-ok`,
  `stub-chatty-json`, `stub-bad-json`, `stub-stream`) and off `response_format.type`.
  Prompt-mode schema jobs are answered by matching `"matching this schema"` in the prompt.
- Run the suite with `-W error::ResourceWarning`; it catches socket leaks in the provider.
- `run_pipeline` seeds stage 1 with a pseudo-result `Result(job="__input__")`; `out["stages"]`
  slices it off with `ran = st["stages"][1:]`. Preserve that when adding slots.
- Cache lives in `.harness/cache/` and is **not** gitignored (the parent repo has no rule
  for it; the dir does not exist yet). Add `.harness/` to `.gitignore` before the first
  commit that runs the CLI, or cache files will land in the repo.
