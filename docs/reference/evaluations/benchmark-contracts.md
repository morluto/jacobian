# Benchmark contracts

[Documentation home](../../index.md) · [Capability surface](../tools.md)

All executable benchmark cases are Harbor tasks under their dataset roots
([`benchmarks/datasets/`](../../../benchmarks/README.md)). The datasets
retain separate claims:

- `agent-workflow-v1` checks fixed workflows and evidence handling;
- `symbolic-coordination-v1` owns the exact polynomial-map coordination pilot;
- `public-reproductions-v1` replays known public mathematical outcomes;
- `research-diagnostics-v1` supports answer-visible case diagnostics;
- `provider-feasibility-v1` reproduces optional-provider pins and outcomes;
- `examples-v1` owns non-comparative tutorial and smoke workflows.

`registry.toml` is the discovery index. A dataset's `suite.toml` owns its
stable policy, while sorted `members/*.toml` records authoritatively bind
canonical task IDs to provenance, assurance, environment, and verifier
contracts. Content-addressed snapshot locks freeze intentional evaluation sets;
publication manifests are generated from locks outside dataset roots.

Dataset identity is a claim boundary: workflow observations, public
reproductions, answer-visible research diagnostics, runtime measurements,
provider feasibility, and examples must not share an interpretation merely
because they use one task format.

The ownership boundary is deliberate. `benchmarks/datasets/` contains
executable Harbor cases and task-owned analysis records, while
`benchmarks/tooling/` contains reusable Harbor infrastructure. Analysis records
may capture discovery context, but they do not duplicate tasks, become Harbor
job input, or enter an agent container.

## Task contract

Tasks use `benchmarks/datasets/<dataset>/<task-id>/` with a maintainer README,
agent-visible instruction and environment, Oracle-only solution, and
verifier-only tests. Mathematical tasks use `mathematical-sciences`; runtime,
provider, and product-surface tasks use `software-systems`.

Every task has frozen agent-visible input, schema 1.4 metadata, an Oracle-only
solution, and a separate clean-room verifier. The common submission envelope
separates the conclusion, task-specific result, assurance, scope, completeness,
digest-bound evidence, optional verification record, and limitations. Unknown
fields fail closed. Task-specific schemas may narrow the result but cannot
weaken the envelope.

## Task and verifier validation

Task and verifier validation is separate from model observation. For an
ordinary leaf change, validate the selected task and its exact Oracle:

```sh
make harbor-check-task DATASET=<dataset-id> TASKS="<task-id>"
make harbor-oracle-task DATASET=<dataset-id> TASKS="<task-id>"
```

The focused commands require explicit task IDs and do not fall back to all
tasks. The full repository gate remains the appropriate check for
`registry.toml`, suite headers and policy, shared tooling, schemas, global
task-ID uniqueness, or other control-plane changes:

```sh
make harbor-check
make benchmark-inventory OUTPUT=/tmp/benchmark-inventory.json
make harbor-oracle DATASET=agent-workflow-v1 FULL=1
```

A task README or a host-side regression under `benchmarks/validation/` changes
documentation or deterministic validation only, so it does not require an
Oracle. Changes to a task's executable input, environment, solution, member
record, or clean-room verifier do require the selected-task Oracle after the
contract check.

The suite module checks that each member ID names a direct Harbor task bundle
and validates the generated task digests. The verifier scores only evidence its
contract authorizes. Wrong mathematical answers, malformed or escaped evidence,
incomplete scope, and false certification receive zero reward. An Oracle answer
does not authorize `VERIFIED`.

Each separate verifier owns its local `tests/verifier_support.py`; Harbor's
whole-task digest binds that copy, so validation does not synchronize it with a
global runtime helper. New tasks inherit the template copy, while shared fixes
are explicit migrations over selected tasks. Use the scoped `harbor-sync`
command only after such a deliberate update. Evidence has no arbitrary byte
cap, but its schema, digest, path, and workspace binding remain mandatory.
Verifier regression fixtures should also prove that malformed submissions do
not crash: exercise booleans where integers are expected, non-finite JSON
numbers, unhashable nested values, wrong-shaped input, and assurance or
protocol failures whose independent diagnostics remain visible. A full Oracle
reward does not replace these negative-path checks.

`TIMEOUT`, `CANCELLED`, `ERROR`, incomplete enumeration, and failure to find a
witness remain non-conclusions. Only operator-authorized independent checkers
may accept `VERIFIED`.

## Reproducible handoff

Record the git tree, suite and task digests, provider/runtime profile, model and
prompt settings, raw trace location, validation actually run, unresolved proof
obligations, and next action. Publishing a local dataset to a Harbor registry
requires separate authorization.

Ordinary executable task additions are leaf-only: the direct task bundle and
its matching `members/<task>.toml` record. They change the prospective suite
digest without rewriting stable suite policy or existing snapshot locks.
Intentional evaluation and publication events create a content-addressed lock
under `benchmarks/snapshots/`; publication manifests are generated under
ignored `dist/harbor/` from that lock.

See [evaluation methods](evaluation-methods.md) for workflow observation,
performance measurement, and interpretation guidance.
