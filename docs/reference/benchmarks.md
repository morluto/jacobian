# Benchmark contracts

[Documentation home](../index.md)

All executable benchmark cases are Harbor tasks under their dataset roots
([`benchmarks/datasets/`](../../benchmarks/README.md)). The six datasets retain
separate claims:

- `agent-workflow-v1` checks fixed workflows and evidence handling;
- `public-reproductions-v1` replays known public mathematical outcomes;
- `research-diagnostics-v1` supports answer-visible case diagnostics;
- `performance-v1` records report-only operational measurements;
- `provider-feasibility-v1` reproduces optional-provider pins and outcomes;
- `examples-v1` owns non-comparative tutorial and smoke workflows.

`registry.toml` is the discovery index. A dataset's `suite.toml` owns its
stable policy, while sorted `members/*.toml` records authoritatively bind
canonical task IDs to provenance, assurance, environment, and verifier
contracts. Content-addressed snapshot locks freeze intentional evaluation sets;
publication manifests are generated from locks outside dataset roots.

## Task contract

Tasks use `benchmarks/datasets/<dataset>/<task-id>/` with a maintainer README,
agent-visible instruction and environment, Oracle-only solution, and
verifier-only tests. Mathematical tasks use `mathematical-sciences`; runtime,
provider, and product-surface tasks use `software-systems`.

The common submission envelope separates the conclusion, task-specific result,
assurance, scope, completeness, digest-bound evidence, optional verification
record, and limitations. Unknown fields fail closed. Task-specific schemas may
narrow the result but cannot weaken the envelope.

Wrong mathematical answers and false certification force reward to zero.
`TIMEOUT`, `CANCELLED`, `ERROR`, incomplete enumeration, and failure to find a
witness remain non-conclusions. Only operator-authorized independent checkers
may accept `VERIFIED`.

See [agent workflow observations](agent-evaluations.md) and
[performance measurements](performance-benchmarks.md) for dataset-specific
interpretation.
