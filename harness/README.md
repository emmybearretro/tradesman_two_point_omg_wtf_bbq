# harness

Multi-model LLM harness over any OpenAI-compatible endpoint. Stdlib only — no
pip install, Python 3.11+.

- **Providers**: base URL + optional key. Works with Ollama, vLLM, LM Studio,
  OpenAI, OpenRouter, LiteLLM, llm-cpp, anything speaking `/v1/chat/completions`.
- **Model selection**: aliases, fallback chains, live discovery from
  `/v1/models`. Models that load later are resolved as *unlisted* and attempted,
  so a chain keeps working as you load models.
- **Jobs**: a task bound to a model chain, with prompt template, sampling params,
  and an output contract (text / JSON / JSON Schema, validated).
- **Pipelines**: stages chained per item; each stage runs on its own model.

## Run

```sh
./harness.py <cmd>            # or: python3 -m harness <cmd>
```

Exit codes: `0` all ok, `1` any item failed, `2` usage/config error.

## Configure

`config.toml` (searched as `./config.toml`, `./harness.toml`,
`$HARNESS_CONFIG`, `$XDG_CONFIG_HOME/harness/config.toml`; `-c PATH` to
override). `${VAR}` and `${VAR:-default}` expand from the environment, so keys
never live in the file.

```toml
[providers.ollama]
base_url = "http://127.0.0.1:11434/v1"
default = true                       # bare model ids land here

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"       # keyless: omit api_key* entirely
strict = true                        # only serve models listed by /v1/models

[providers.gateway]
base_url = "${LLM_GATEWAY_URL:-http://127.0.0.1:8080/v1}"
api_key = "${LLM_GATEWAY_KEY}"       # empty env -> no Authorization header
headers = { "x-trace" = "harness" }
extra = { temperature = 0.2 }        # provider defaults, per-job params win
```

Provider keys: `base_url` (required), `api_key`, `api_key_env`, `timeout` (120),
`max_retries` (2), `strict` (false), `headers`, `extra`, `default`.

`strict = true` refuses ids the provider does not list — right for a gateway
with a fixed serving list. Default `false` means an unlisted id is a real
attempt, flagged `(unlisted)` in output.

## Models

```toml
[aliases]
fast = "ollama/llama3.2:3b"
smart = ["openai/gpt-4o", "gateway/claude-sonnet-4-5", "ollama/qwen2.5-coder:7b"]
workhorse = "smart"                  # aliases may point at aliases
```

A model spec resolves to a fallback chain. Forms: alias, `provider/model`,
bare `model` (provider that serves it, else fuzzy prefix match, else default
provider), or a list. Chains are tried left to right; a target is skipped when
its provider is strict and does not serve it, and the next target runs. A
provider fault (429/5xx/timeout) is retried with backoff, then falls through.

```sh
./harness.py models            # providers + served models, per provider
./harness.py models --refresh  # force re-discovery (TTL 300s)
./harness.py ping              # reachability + latency
./harness.py jobs              # resolved chain per job, with skip notes
```

## Jobs

```toml
[jobs.triage]
model = "fast"
system = "Label the input. Reply with JSON only."
template = "Text: {input}"           # {field} reads item fields; missing -> error
format = "schema"                    # text | json | schema
temperature = 0
max_tokens = 512
max_workers = 4                      # concurrent items
cache = true                         # disk cache keyed on model+messages+params
schema = { type = "object", required = ["category"],
           properties = { category = { type = "string", enum = ["bug", "feature"] } } }
```

`format = "json"` requests `response_format: json_object` and parses;
`format = "schema"` sends the schema as `response_format: json_schema` and
validates the reply locally (type/required/enum/items/const/min/max). A reply
wrapped in prose or a ``` fence is extracted before parsing. Parse or schema
failure is retried once on the next model in the chain.

```sh
./harness.py ask -j triage "crash on startup"
./harness.py ask -m smart -p "hi" --stream
./harness.py run triage -i items.jsonl -o out.jsonl -w 8
./harness.py run triage -p one -p two --json
echo a b c | ./harness.py run summarize -i -
```

Items: `-p` inline, `-i FILE` (`.json` array, `.jsonl`, or plain lines), `-i -`
for stdin. A dict item's `key` labels the row; other fields feed `{field}`.

## Pipelines

```toml
[pipelines.triage-then-summarize]
fail_fast = true
stages = [
  { job = "triage" },
  { job = "summarize", input = "json" },   # text | parsed | json | field:<name>
]
```

`input` selects what the next stage sees; `keep = false` drops upstream item
fields from the handoff; `model = "..."` overrides the stage's chain.

```sh
./harness.py pipelines
./harness.py pipe triage-then-summarize -i items.jsonl -o out.jsonl
```

Each output row carries the final result plus a `stages` array with every
stage's model, latency, usage, and parsed value.

## Output

`--json` emits one record per item: `ok`, `job`, `key`, `target`
(`provider/model`), `tried`, `text`, `parsed`, `usage`, `latency`, `cached`,
`attempts`, `error`. `--json` on `ask` prints a single record.

## Cache

`.harness/cache/<sha256>.json`, keyed on model chain + messages + params +
format + schema. Change the model, prompt, params, or schema and the key
changes. `--no-cache` bypasses; `cache = false` per job.

## Tests

```sh
python3 tests/run_tests.py
```

68 tests against a local OpenAI-compatible stub: discovery, alias/fuzzy/strict
resolution, key sending, retry/fallback, SSE streaming, JSON extraction, schema
validation, cache invalidation, pipeline chaining, CLI exit codes.
