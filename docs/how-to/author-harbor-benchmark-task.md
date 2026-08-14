# Author a Harbor benchmark task

[Documentation home](../index.md)

Executable benchmark cases live directly under their Harbor dataset root as
`benchmarks/datasets/<dataset>/<task-id>/`. Choose a globally unique flat task
ID; keep domain, field, provenance, and evaluation classification in
`task.toml` metadata. Add one member fragment at
`benchmarks/datasets/<dataset>/members/<task-id>.toml`. Start from
`benchmarks/templates/task/` and `benchmarks/templates/member.toml`, then
choose the dataset whose claim matches the case. The ordinary change must stay
within those two leaf paths; do not bump suite policy or update historical
evaluation files.

Freeze agent-visible input and the strict submission schema under
`environment/`. Keep expected answers and solution material under `solution/`,
and the clean-room verifier plus fixtures under `tests/`. Harbor injects
`solution/` only for the Oracle and uploads `tests/` only for verification, so
an environment Dockerfile must never copy either directory. The member record
binds stable task identity, classification, provenance, assurance ceiling,
provider, environment profile, verifier contract, and evaluation owner.

Use the repository's named environment profiles. Every `FROM` must match its
profile's immutable image digest. Standard task Dockerfiles copy only genuine
task input and schemas; they do not run `apt-get`. Provider-specific tasks may
install the dependencies that define that provider experiment. Use Harbor's
native `public`, `no-network`, or `allowlist` task network policy. Proxy-backed
operation belongs in the Harbor job composition and does not select an image.
For a shared checker or external executable, keep the agent and verifier images
separate and pin the published `@sha256:` reference; see
[reusable evaluation images](../reference/evaluations/benchmark-contracts.md#reusable-evaluation-images).

Run `make harbor-plan BASE=origin/main`, inspect the prospective digest and
exact changed-task lane, then validate and Oracle only the task being authored:

```sh
make harbor-check-task DATASET=<dataset-id> TASKS="<task-id>"
make harbor-oracle-task DATASET=<dataset-id> TASKS="<task-id>"
```

The focused gate requires an explicit dataset and one or more task IDs; it does
not silently expand to the full corpus. Use `make harbor-check` for changes to
shared Harbor contracts, schemas, the registry, suite policy, or other
control-plane files. Reserve `make harbor-check-all` for shared verifier-harness
changes or an intentional full local reproduction. For
`mathematical-benchmarks-v1` pytest regressions, put verifier
attack cases in
`benchmarks/validation/mathematical_benchmarks_v1/test_<task_id_with_underscores>.py`
rather than a shared dump; see the Harbor skill's validation regression layout
note. Control/treatment observation jobs are committed,
three-attempt reproducibility fixtures, but running `make agent-eval ...
EVAL_EXECUTE=1` is an explicit operator-run evidence exercise, not a task
authoring or pull-request gate.

Task README edits and host-side regression tests under
`benchmarks/validation/` do not change executable task inputs and do not need
an Oracle. Changes to the task environment, instructions, solution, member
record, dependencies, image, or clean-room verifier do: rerun the exact
selected-task Oracle after validation.

Do not create a snapshot for every task addition. An operator creates a new
content-addressed snapshot only when intentionally freezing an evaluation or
publication set. Publication `dataset.toml` files are generated from that lock
under ignored `dist/harbor/`; they are never committed in the dataset root.
Use `make benchmark-snapshot DATASET=<dataset-id>` from a clean pre-lock tree,
then validate and publish with `make benchmark-snapshot-validate LOCK=<lock>`
and `make benchmark-publish LOCK=<lock>`.

The task-local `tests/verifier_support.py` is authoritative for that task and
is bound by Harbor's task digest. New tasks receive the generic helper from
`benchmarks/templates/task/`; existing tasks are not silently upgraded. If
you change `tests/verifier.py`, update only that task's checksum label with
`make harbor-sync DATASET=<dataset-id> TASKS="<task-id>"`. The command requires
both selectors and never rewrites unrelated tasks; `harbor-check-task` is
strictly read-only.

Do not add task symlinks, aliases, or a second fixture home. The task
README is maintainer context and is not injected into a trial. Instructions
describe the requested outcome without prescribing Jacobian operations or a
research strategy.

Verifier attack tests should cover malformed and unknown fields, wrong answers,
scope and completeness mismatches, forged or escaped evidence, digest mismatch,
false assurance, booleans in integer fields, non-finite numbers, and unhashable
nested values. Bound regular submissions and visible/frozen inputs before
reading them to keep malformed inputs from exhausting the verifier. Do not
impose an arbitrary byte limit on otherwise valid benchmark evidence; validate
its schema, digest, exact path, and workspace binding instead. A task may
accept `VERIFIED` only when an
operator-authorized checker independently binds the exact claim and evidence.
After changing a task contract, verifier, dependency, or image, recompute its
prospective Harbor digest and rerun its exact Oracle.
