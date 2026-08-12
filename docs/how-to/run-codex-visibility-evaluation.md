# Run the MCP visibility evaluation

Use this operator-run diagnostic to compare a control with no Jacobian against
a treatment that exposes Jacobian MCP only. It measures whether Codex discovers
and uses the fixed tool surface; it does not grade mathematical prose or replace
Harbor correctness evaluations.

Start with a deployed or local Streamable HTTP endpoint. Run the control with
no Jacobian MCP configuration:

```sh
make codex-visibility VISIBILITY_EXECUTE=1 \
  VISIBILITY_MODEL=gpt-5.6-sol \
  VISIBILITY_OUTPUT=benchmarks/results/visibility-control
```

Run the treatment with the endpoint and no Jacobian Skill:

```sh
make codex-visibility VISIBILITY_EXECUTE=1 \
  VISIBILITY_MCP_URL=http://127.0.0.1:8000/mcp \
  VISIBILITY_MODEL=gpt-5.6-sol \
  VISIBILITY_OUTPUT=benchmarks/results/visibility-mcp
```

Set `VISIBILITY_TOOL_MODE=unified_exec` to reproduce Codex Code Mode's nested
tool dispatch, as used by the Harbor adapter. Code Mode exposes nested methods
through `tools`, so measure its model-visible context directly rather than
assuming it matches native MCP presentation.

The runner refuses to overwrite an existing output directory. Each result
binds the prompt-suite digest, Git revision, Codex version, model, reasoning
effort, evaluator and telemetry-parser digests, MCP server metadata, tool
schemas and descriptions, and catalog digest. Raw JSONL and stderr are retained
with SHA-256 digests.

Each Codex process receives a temporary isolated `HOME` and `CODEX_HOME` seeded
with authentication only; user config, plugins, rules, memories, and ambient
skills are not copied. Before model execution, the runner renders Codex's actual
model-visible prompt, records the normalized skill names, sources, and digest,
and fails if any file-backed skill escapes the isolated homes or workspace.

The suite covers matrix, integer, polynomial, and independent-checker outcomes.
It also includes negative cases whose `ABSTAIN` expectation permits no Jacobian
tool calls or resource reads. Observations report discovery, exact inspection,
invocation, completion, abstention, unexpected operations, and independently
bound verification evidence.

Case contracts gate only required outcome operations and verification evidence.
Optional `diagnostic_capability_ids` record whether an operation was discovered,
attempted, or completed without requiring one fixed tool sequence for success.
`acceptable_output_outcomes` can instead require substantive fields from any one
of several completed atomic operations; this scores structured mathematical
output rather than requiring a redundant operation or grading answer prose.

Duration and token counts are observational. MCP calls, model-visible bytes,
wire bytes, shell calls, errors, cached input, and uncached input remain
separate diagnostics rather than proxies for mathematical correctness.
The report also records empty-payload probes, failed operation attempts, exact
repeated errors, and per-case rates across repetitions. These recovery metrics
remain observational and never change a mathematical or verification verdict.

To run the fixed Lean usability observation suite, select it explicitly:

```sh
make codex-visibility VISIBILITY_EXECUTE=1 \
  VISIBILITY_CASES=benchmarks/config/lean-usability-v1.json \
  VISIBILITY_MCP_URL=http://127.0.0.1:8000/mcp \
  VISIBILITY_MODEL=gpt-5.6-sol \
  VISIBILITY_OUTPUT=benchmarks/results/lean-usability
```

That suite observes premise retrieval, fresh and continuation proof-state use,
term application, independent checking, Mathlib declaration discovery,
runtime-identity reporting, and one conceptual abstention. Formal-intermediate
operations are diagnostic observations rather than required proof strategies.
It is a public usability regression, not held-out or causal evidence.

For Harbor ATIF trajectories, measure the projection directly:

```sh
make codex-tool-context \
  TRAJECTORIES="benchmarks/results/<job>/<trial>/agent/trajectory.json" \
  LABEL=mcp-treatment \
  OUTPUT=benchmarks/results/tool-context-treatment.json
```

For a comparison, hold the suite digest, Codex version, model, reasoning
effort, budgets, and repetition count fixed. The only intervention is Jacobian
MCP availability. Use multiple repetitions and report each cue level and
expectation separately. Public-suite results are regression observations, not
held-out causal evidence.
