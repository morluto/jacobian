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

The runner refuses to overwrite an existing output directory. Each output binds
the prompt-suite digest, Git revision, Codex version, model, reasoning effort,
skill digest, evaluator and telemetry-parser digests, MCP server metadata, tool
schemas and descriptions, and catalog digest. Raw JSONL and stderr are retained
with SHA-256 digests.

The v1 suite separates an explicit integration canary from affordance and latent
cases. Per-run observations report discovery, exact contract inspection,
invocation, completion, and independently bound `VERIFIED` evidence. Shell calls,
tool errors, tokens, MCP wire bytes, model-visible bytes, and logical payload bytes
are independent diagnostics rather than proxies for mathematical correctness.
The cumulative Codex input-token count may grow much faster than MCP payload
bytes because each later model turn includes earlier tool results; compare both
dimensions rather than treating adoption alone as a win.

For an A/B claim, hold the suite digest, Codex version, model, reasoning effort,
MCP catalog, budgets, and repetition count fixed. Change only the visibility
condition, use multiple repetitions, and report each cue level separately. Public
v1 results are regression evidence, not held-out causal evidence.
