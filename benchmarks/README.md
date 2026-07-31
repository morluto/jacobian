# Benchmark and evaluation artifacts

This tree holds **validation and observation surfaces**, not the product API.
Mathematical correctness gates live in contract and domain tests; agent
workflow observation uses Harbor; research challenges are answer-visible
diagnostics only.

## Ownership map

| Path | Artifact class | What it proves | Authoritative for |
| --- | --- | --- | --- |
| [`regression-v1/`](regression-v1/) | Harbor workflow dataset | Offline Oracle contract + Jacobian observation on a release-frozen task set | Agent workflow observation (v1); `dataset.toml` is refreshed only by release PRs |
| [`capability-evaluations/v1/`](capability-evaluations/v1/) | Gap ledger + C1/C2 evaluation scaffold | Public workflow classification and fail-closed comparison design | Discovery handoffs and future held-out configuration; **not** a model-performance claim |
| [`research/`](research/) | Public research challenges + runner | Answer-visible composition diagnostics under no-retrieval policy | Capability discovery handoffs; **not** held-out model scores |
| [`reproductions/`](reproductions/) | Public reproduction fixtures | Exact public case replay in composition/provider tests | Regression of known public mathematical episodes |
| [`examples/`](examples/) | Pilot / documentation cases | Illustrative inputs only | Docs and local smoke |
| [`performance/`](performance/) | Operational microbenchmarks | Store/dispatch/search speed and resource cost | Performance baselines (see [performance benchmarks](../docs/reference/performance-benchmarks.md)) |
| [`provider_spikes/`](provider_spikes/) | Optional backend spikes + pins | Feasibility and pin fidelity for unmerged providers | Contributing spikes; **not** product surface |
| `results/` | Local Harbor / runner outputs | Run artifacts (gitignored) | Host-local only |

### Frozen path

**`benchmarks/regression-v1/` is path-stable.** Job JSON, compose files,
Makefile targets, skills, and external runners hardcode this location. Move it
only in a dedicated migration that updates every consumer and re-runs
`make harbor-check` plus Oracle.

### What does not belong here

- Production capability adapters (`src/jacobian/`, `src/jacobian/domains/`)
- Independent checkers (`src/jacobian_checkers/`)
- Runtime services such as `EvaluationService` (plugin batch evaluation lives
  under the package runtime, not under Harbor research tooling)

Agent-eval helpers that are not Harbor tasks live in `src/jacobian/eval/`
(telemetry parsing, graph oracles).

## Hierarchy of claim strength

From strongest operational gate to loosest diagnostic:

1. Contract / adversarial conformance (tests)
2. Public mathematical scenarios and reproductions
3. Harbor `regression-v1` Oracle + observation
4. Performance microbenchmarks
5. Public research challenges (answer-visible)
6. Provider spikes (optional backends)

A famous open problem is never a substitute for layers 1–3.

## Commands

```sh
# Harbor task validation + Oracle contract
make harbor-check
make harbor-oracle

# Release PR only, after all task changes have landed
make harbor-release-sync
make harbor-release-oracle

# Core performance microbenchmark
make bench-core

# Public research challenge plan (no model run)
uv run python benchmarks/research/runner.py --challenge jcb-postdoc-014

# Provider spike (example)
uv run python benchmarks/provider_spikes/nauty_provider_spike.py
```

## Related docs

- [Reference benchmarks](../docs/reference/benchmarks.md)
- [Agent evaluations](../docs/reference/agent-evaluations.md)
- [Performance benchmarks](../docs/reference/performance-benchmarks.md)
- [Capability workflow evaluations](../docs/reference/capability-workflow-evaluations.md)
- [Harbor benchmarks skill](../.agents/skills/harbor-benchmarks/SKILL.md)
