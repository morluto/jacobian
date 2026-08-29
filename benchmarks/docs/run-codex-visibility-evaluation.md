# Run the MCP visibility evaluation

[Benchmark home](../README.md)

This opt-in diagnostic measures distinct MCP adoption stages without grading
answer prose. The general suites compare no-Jacobian and Jacobian conditions.
The frozen `direct-mcp-agent-adoption-v1.json` suite instead compares these
Jacobian surfaces:

| Arm | Visible MCP tools | Question |
| --- | --- | --- |
| `legacy` | `math.find`, `math.run` | Can the agent discover and execute through the generic pair? |
| `direct` | catalog operation tools | Can the agent select, invoke, and compose typed tools directly? |
| `direct_find` | direct tools plus `math.find` | Does semantic vocabulary search improve selection? |
| `find_only` | `math.find` | Which mathematical vocabulary is discoverable independently of execution? |

The frozen corpus contains straightforward, synonym, postcondition,
structurally ambiguous, and multi-operation cases. Its semantic-discovery
cases are scored on descriptions returned by `math.find`; they do not require
an operation invocation.

Freeze the prompt suite, model, environment, and endpoint configuration before
running either arm. Store the raw outputs with the evaluation's own metadata
and interpret them as research evidence only.

Run the deterministic catalog controls first:

```sh
make direct-mcp-catalog-eval \
  DIRECT_MCP_EVAL_OUTPUT=tmp/direct-mcp-catalog-evaluation.json
```

They exercise the production in-memory MCP server and record catalog/schema
coverage, serialized definition bytes, `tools/list` latency, `math.find`
ranking, direct and `math.run` execution parity, and exact typed composition.
They intentionally report model selection, deferred client search, and exact
per-task loaded-definition bytes as unmeasured.

For agent trials, start the anonymous loopback server, then select an arm and
an output directory:

```sh
uv run python -m jacobian.mcp.remote_cli \
  --allow-anonymous --host 127.0.0.1 --port 8765 --path /mcp

make codex-visibility \
  VISIBILITY_EXECUTE=1 \
  VISIBILITY_CASES=benchmarks/config/direct-mcp-agent-adoption-v1.json \
  VISIBILITY_MCP_URL=http://127.0.0.1:8765/mcp \
  VISIBILITY_MODEL=<model> \
  VISIBILITY_SURFACE_ARM=direct \
  VISIBILITY_CASES_SELECTED="straightforward-determinant alternate-term-sandpile-group complete-subset-sum-profile structured-polynomial-gcd canonical-cnf-composition" \
  VISIBILITY_REPETITIONS=2 \
  VISIBILITY_OUTPUT=tmp/direct-mcp-agent-direct-r2
```

The `direct` arm selects the five execution cases; the two semantic-discovery
cases belong in a separate `find_only` run:

```sh
make codex-visibility \
  VISIBILITY_EXECUTE=1 \
  VISIBILITY_CASES=benchmarks/config/direct-mcp-agent-adoption-v1.json \
  VISIBILITY_MCP_URL=http://127.0.0.1:8765/mcp \
  VISIBILITY_MODEL=<model> \
  VISIBILITY_SURFACE_ARM=find_only \
  VISIBILITY_CASES_SELECTED="semantic-sandpile-synonym semantic-chip-firing-neighborhood" \
  VISIBILITY_REPETITIONS=2 \
  VISIBILITY_OUTPUT=tmp/direct-mcp-agent-find-only-r2
```

`VISIBILITY_CASES_SELECTED="case-a case-b"` restricts a run without changing
the frozen suite. Each fresh Codex process receives an isolated home and user
configuration, and each report binds the suite, surface, runner, telemetry
parser, model, client version, and visible skill surface by digest.

The report distinguishes the complete server definition size from the exact
configured arm size. Current Codex JSONL telemetry does not expose which tool
definitions client-managed search actually loaded into model context. The
field `exact_loaded_tool_definition_bytes` therefore remains `null`; token
counts or invoked-tool definition sizes must not be presented as a substitute.
