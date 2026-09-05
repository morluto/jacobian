---
name: harbor-benchmarks
description: Author, package, validate, or run mathematical evaluations as Jacobian Harbor datasets.
---

# Harbor Benchmarks

Build a bounded mathematical experiment with a hidden, replayable verifier.
Jacobian availability is an experimental condition, not a prerequisite for task
admission. Task correctness depends on the submitted mathematics, not the
agent's tool choices or transcript.

## Choose the artifact

- Put meaningful agent-facing mathematical tasks with replayable outcomes in
  Harbor, including capability gaps and tool-free baselines.
- Put provider/environment feasibility spikes under `tests/fixtures/providers/`.
- Keep public known-answer regressions and human-reviewed proof corpora separate
  from held-out comparative scoring.

For submission design, mathematical predicates, scoring, or verifier repairs,
use `verifier-evaluations`. Ordinary packaging and execution do not require that
skill. A task's instruction and schema must expose its complete protocol while
keeping solutions, Oracle fixtures, verifier code, and host caches hidden.

## Package and validate

Each task is a direct dataset child with one member record:

```text
benchmarks/datasets/<dataset>/<task-id>/
benchmarks/datasets/<dataset>/members/<task-id>.toml
```

Use frozen offline inputs, pinned images and dependencies, and an explicit
execution profile. Verifier Dockerfiles build from `tests/`; avoid parent
`COPY`, host paths, floating tags, and symlinks. For the affected contract, use
[benchmark contracts](../../../benchmarks/docs/benchmark-contracts.md).
For local commands and planner selection, use the Harbor section of
[CONTRIBUTING.md](../../../CONTRIBUTING.md#harbor-and-oracle-validation).

```sh
make harbor-plan BASE=origin/main
make harbor-prepare-task DATASET=<dataset> TASKS="<task-id>"
make harbor-validate-task DATASET=<dataset> TASKS="<task-id>"
```

Preparation formats selected task Python and refreshes contracts/checksums;
validation runs the selected static, host, and Oracle checks without a model
agent. Run the selected verifier's behavioral attacks when its contract changes.
Report any deferred affected Oracle coverage.

For task-local support migrations, older contracts, checksum updates, or an
intentional snapshot, read [packaging details](references/packaging.md).
Do not create a snapshot merely because a task contract changed.

Complete the requested task preparation and validation, repairing failures
caused by the change within the authorized scope. Report the command, task
digest, selected scope, Oracle result, and remaining evidence gaps. Distinguish
local validation from a causal model comparison; model calls and external
publication require the applicable user authorization.
