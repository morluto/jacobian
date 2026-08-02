# Capability workflow evaluations

[Documentation home](../index.md)

Jacobian packages executable evaluation cases into the six Harbor datasets
listed in [`benchmarks/README.md`](../../benchmarks/README.md). Immutable
snapshot locks under `benchmarks/snapshots/` define frozen evaluation and
publication sets.
Dataset identity is a claim boundary: workflow observations, public
reproductions, answer-visible research diagnostics, runtime measurements,
provider feasibility, and examples must not share an interpretation merely
because they use one task format.

The ownership boundary is deliberate. `benchmarks/datasets/` contains
executable Harbor cases and task-owned analysis records, while
`benchmarks/tooling/` contains reusable Harbor infrastructure. Analysis records
may capture discovery context, but they do not duplicate tasks, become Harbor
job input, or enter an agent container.

## Task and verifier validation

Every task has frozen agent-visible input, schema 1.4 metadata, an Oracle-only
solution, and a separate clean-room verifier. The repository suite module
validates `registry.toml`, suite headers, and member fragments, checks that
member IDs resolve to direct Harbor task bundles, computes Harbor-native
digests, checks global task-ID uniqueness and visibility, validates every
submission schema and static Oracle submission, and emits a content-bound
inventory:

```sh
make harbor-check
make benchmark-inventory OUTPUT=/tmp/benchmark-inventory.json
make harbor-oracle DATASET=agent-workflow-v1
```

The verifier scores only evidence its contract authorizes. Timeout,
cancellation, errors, incomplete enumeration, and missing witnesses are
non-conclusions. An Oracle answer does not authorize `VERIFIED`.

## Workflow observations and diagnostics

Use the fixed `agent-workflow-v1` tasks for Jacobian workflow observations:

```sh
make agent-eval DATASET=agent-workflow-v1 EVAL_EXECUTE=1
```

Normalize each condition with `make agent-eval-validate`, passing a
`RUNTIME_SNAPSHOT` that binds the immutable benchmark snapshot ID and pinned
Harbor version, then compare the two
evidence files with `make agent-eval-compare`. The comparator rejects unmatched
task repetitions or drift in task digests, prompts, models, budgets, and job
configuration. It reports correctness and assurance separately and marks
small samples as descriptive. Public suite comparisons remain workflow
observations.

Held-out C1/C2 runs use `.github/workflows/heldout-benchmarks.yml`. A protected
GitHub environment assumes a read-only S3 role through OIDC, validates the
private manifest and archive before extraction, renders both conditions from
one frozen specification, and uploads only non-Oracle results. The pilot fixes
three tasks and three repetitions; a decision run requires at least five tasks
and five repetitions. Neither report automatically authorizes a causal claim.

Use `research-diagnostics-v1` only for answer-visible diagnostic runs. Its
public source answers and Oracle summaries remain hidden from the agent
container at runtime, but their public availability permanently disqualifies
the dataset from held-out model claims.

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
