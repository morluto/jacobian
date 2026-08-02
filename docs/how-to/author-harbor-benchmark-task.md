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

Run `make harbor-plan BASE=origin/main`, inspect the prospective digest and
exact changed-task lane, and run `make harbor-check` followed by
`make harbor-oracle DATASET=<dataset-id> TASKS="<task-id>"`.

Do not create a snapshot for every task addition. An operator creates a new
content-addressed snapshot only when intentionally freezing an evaluation or
publication set. Publication `dataset.toml` files are generated from that lock
under ignored `dist/harbor/`; they are never committed in the dataset root.
Use `make benchmark-snapshot DATASET=<dataset-id>` from a clean pre-lock tree,
then validate and publish with `make benchmark-snapshot-validate LOCK=<lock>`
and `make benchmark-publish LOCK=<lock>`.

Do not add task symlinks, aliases, or a second fixture home. The task
README is maintainer context and is not injected into a trial. Instructions
describe the requested outcome without prescribing Jacobian capabilities or a
research strategy.

Verifier attack tests should cover malformed and unknown fields, wrong answers,
scope and completeness mismatches, forged or escaped evidence, digest mismatch,
and false assurance. A task may accept `VERIFIED` only when an
operator-authorized checker independently binds the exact claim and evidence.
After changing a task contract, verifier, dependency, or image, recompute its
prospective Harbor digest and rerun its exact Oracle.
