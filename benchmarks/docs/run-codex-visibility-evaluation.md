# Run the direct MCP adoption evaluation

[Benchmark home](../README.md)

This external diagnostic separates availability, discovery, selection, and
execution for Jacobian's direct typed MCP surface. It does not score a tool call
as success: each positive case in
[`direct-mcp-adoption-v1.json`](../config/direct-mcp-adoption-v1.json) requires
the expected operation or operations to complete and a typed mathematical
output to match frozen required fields and values.

The treatment assumes a client implementing
[deferred MCP tool search](https://developers.openai.com/api/docs/guides/tools-tool-search):
the server still exposes its exact immutable surface, while the client initially
presents only the server description and loads matching tool definitions at
runtime. Do not interpret a low model-visible byte count as a small MCP
`tools/list` response; the evaluator records both separately.

## Falsifiable questions

With the prompt suite, model snapshot, reasoning effort, Codex version, tool
mode, attempt count, client timeout, Jacobian revision, endpoint configuration,
and egress state fixed:

1. Does replacing generic `math.find` → `math.run` execution with deferred
   direct typed operations preserve or improve valid first execution and final
   typed mathematical outcomes without increasing retries, visible bytes,
   latency, or false-positive calls on abstention cases?
2. On synonym-heavy and postcondition-ambiguous cases, does adding `math.find`
   to the same direct surface improve discovery or final outcomes beyond client
   tool search alone?
3. In unified-exec mode, can one agent compose producer and consumer operations
   directly without reconstructing an operation-ID/payload envelope?

Do not combine those conclusions. A direct call proves execution, not natural
discovery; a correct answer without a call remains a valid task outcome; and a
call with the wrong mathematical result is a failure.

## Frozen arms

Use fresh Codex contexts and the same selected cases for each arm. Positive
prompts require use of available Jacobian tools without naming an operation ID;
otherwise a model may reasonably solve small canaries itself, which measures
selection preference rather than tool-search precision.

| Arm | MCP surface | Purpose |
| --- | --- | --- |
| A | The parent revision exposing `math.find` and `math.run` | Generic execution control |
| B | Direct operations plus `math.find` | Direct execution treatment with vocabulary search |
| C | The same direct operations without `math.find` | Isolate client tool search from Jacobian vocabulary search |

Run each arm once with `--tool-mode direct` and once with
`--tool-mode unified_exec`. Use at least two repetitions or one independently
varied follow-up case before claiming a stable adoption effect.

The evaluator records the complete server instructions and tool definitions,
catalog digest and operation IDs, tool-list definition bytes and latency,
Codex version and relevant feature state, prompts, raw JSONL, retries, MCP bytes,
tokens, elapsed time, operation inputs and typed outputs. It fails closed unless
the evaluated Codex build reports deferred MCP tool search enabled.

Current Codex JSONL does not expose the exact set of definitions loaded by its
internal tool-search step. The report therefore records the exact server-surface
count and an evidence-backed lower bound—the number of distinct tools actually
invoked—while leaving the exact loaded-schema count null. Do not substitute the
surface count, invocation count, or token count for an unobserved exact value.

## Start treatment endpoints

From the treatment revision, start the two local-only stateless endpoints in
separate terminals:

```sh
uv run --locked python -m benchmarks.tooling.direct_mcp_server --port 8011 --with-math-find
uv run --locked python -m benchmarks.tooling.direct_mcp_server --port 8012
```

Use an independently deployed parent revision for arm A. Do not change the
server, model, prompt suite, or client configuration between repetitions within
one arm.

## Run one arm

Model execution is explicit and may consume paid usage:

```sh
uv run --locked python -m benchmarks.tooling.codex_visibility \
  --execute \
  --cases benchmarks/config/direct-mcp-adoption-v1.json \
  --mcp-url http://127.0.0.1:8011/mcp \
  --model MODEL_SNAPSHOT \
  --reasoning-effort high \
  --tool-mode direct \
  --repetitions 2 \
  --timeout-seconds 900 \
  --output tmp/direct-mcp-arm-b-direct
```

Use a new output directory for every arm and tool mode. An outer timeout only
aborts the evaluation command; it is not a mathematical outcome. Preserve every
report and raw transcript before comparing arms.

## Interpret the result

Compare final typed outcome satisfaction first, then availability and
discovery, valid first execution, recovery, false-positive calls, tokens,
model-visible MCP bytes, total MCP wire bytes, latency, and cost. Attribute a
failure to the smallest observed stage. Retain `math.find` only if arm B shows a
repeated vocabulary-discovery or outcome benefit over arm C that is not merely
an extra call. Keep the result as evaluation evidence outside product
documentation; it does not by itself change an operation's mathematical
contract.
