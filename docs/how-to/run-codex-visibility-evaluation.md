# Run the Codex visibility evaluation

Use this operator-run diagnostic to measure whether Codex notices Jacobian's MCP
affordances before a capability ID or Jacobian-specific workflow is supplied. It
records adoption and transport cost; it does not grade mathematical answer prose
or replace the Harbor correctness evaluations.

Start with a deployed or local Streamable HTTP endpoint, then run a control:

```sh
make codex-visibility VISIBILITY_EXECUTE=1 \
  VISIBILITY_MCP_URL=http://127.0.0.1:8000/mcp \
  VISIBILITY_MODEL=gpt-5.6-sol \
  VISIBILITY_OUTPUT=benchmarks/results/visibility-control
```

Run the treatment with the repository's thin Codex skill:

```sh
make codex-visibility VISIBILITY_EXECUTE=1 \
  VISIBILITY_MCP_URL=http://127.0.0.1:8000/mcp \
  VISIBILITY_MODEL=gpt-5.6-sol \
  VISIBILITY_SKILL=.agents/skills/jacobian-math \
  VISIBILITY_OUTPUT=benchmarks/results/visibility-skill
```

Set `VISIBILITY_TOOL_MODE=unified_exec` to reproduce Codex Code Mode's nested
tool dispatch, as used by the Harbor adapter. This matters for cost testing:
Code Mode exposes nested methods through `tools`, and a model can otherwise
print matching entries from `ALL_TOOLS` into its own context before calling
them.

The runner refuses to overwrite an existing output directory. Each output binds
the prompt-suite digest, Git revision, Codex version, model, reasoning effort,
skill digest, evaluator and telemetry-parser digests, MCP server metadata, tool
schemas and descriptions, and catalog digest. Raw JSONL and stderr are retained
with SHA-256 digests.

The default v2 suite covers matrix, integer, polynomial, and independent
verification outcomes. It also includes negative cases whose `ABSTAIN`
expectation permits no Jacobian MCP tool calls or resource reads. The v1 suite
remains available by setting
`VISIBILITY_CASES=benchmarks/config/codex-visibility-v1.json`.

Per-run observations report discovery, exact contract inspection, invocation,
completion, discovery-free invocation, abstention, unexpected capabilities, and
independently bound `VERIFIED` evidence. Each run also records an
`elapsed_seconds` wall-clock duration measured with a monotonic clock around the
Codex command invocation, and the summary exposes a `duration_totals.elapsed_seconds`
sum. Duration is observational only: it does not affect contract satisfaction,
reward, timeout policy, or mathematical assurance. Single-run wall time is noisy;
paired or repeated comparisons are required before drawing an efficiency
conclusion. Summary cost totals separate cached and uncached input tokens and
include MCP calls and model-visible bytes. Shell calls, tool errors, token
counts, MCP wire bytes, and logical payload bytes remain independent diagnostics
rather than proxies for mathematical correctness. The cumulative Codex
input-token count may grow much faster than MCP payload bytes because each later
model turn includes earlier tool results; compare both dimensions rather than
treating adoption alone as a win.

For Harbor ATIF trajectories, measure that projection directly rather than
inferring it from token totals:

```sh
make codex-tool-context \
  TRAJECTORIES="benchmarks/results/<job>/<trial>/agent/trajectory.json" \
  LABEL=skill-treatment \
  OUTPUT=benchmarks/results/tool-context-treatment.json
```

The report counts `exec` source that references `ALL_TOOLS`, binds the matching
observation bytes by tool-call ID, and reports cached and uncached prompt-token
medians separately. Missing call-ID bindings are reported rather than silently
treated as measured zero-byte projections. It is a client-behavior diagnostic,
not a correctness score or proof that a prompt change caused the observed
difference.

For an A/B claim, hold the suite digest, Codex version, model, reasoning effort,
MCP catalog, budgets, and repetition count fixed. Change only the visibility
condition, use multiple repetitions, and report each cue level and expectation
separately. Public suite results are regression evidence, not held-out causal
evidence.
