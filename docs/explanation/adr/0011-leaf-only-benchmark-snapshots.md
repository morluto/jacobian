# Leaf-only benchmark membership and immutable snapshots

Status: Accepted, pre-stable

## Context

ADR 0008 made Harbor task directories canonical, but retained one generated
`dataset.toml` and copied verifier support in every task. Adding one task then
rewrote shared manifests, suite versions, research records, and CI inventories.
Unrelated task pull requests conflicted and invalidated each other's digests.

Harbor already treats direct task directories as its authoring unit. Registry
publication manifests are useful only after a corpus is intentionally frozen;
they do not need to be mutable authoring inputs.

## Decision

`suite.toml` contains stable policy and defaults. One authoritative
`members/<task-id>.toml` record binds every direct task directory to its stable
identity, classification, provenance, assurance ceiling, provider,
digest-resolved environment profile, verifier contract, and evaluation owner.
The member filename, task ID, task name, and direct directory must agree.

Dataset roots never contain a committed `dataset.toml`. An intentional
evaluation or publication creates a canonical JSON lock under
`benchmarks/snapshots/<suite>/<digest>.lock.json`. Its content address binds the
suite header, ordered Harbor task digests, Harbor version, resolved image and
verifier-runtime digests, source tree, split, and evaluation configuration.
Publication manifests are deterministic outputs under ignored
`dist/harbor/<suite>/<snapshot>/`.

Snapshot creation runs against a clean, pre-lock Git tree. This avoids a
self-reference between a lock and the tree that contains it: the lock pins the
benchmark source tree being frozen, while the subsequent snapshot commit adds
the lock and any reports that cite its content address.

Historical locks and evaluation records retain their identity when current
membership changes. Current-tree reproduction is an explicit check, separate
from validating a lock's intrinsic schema and content address.

Pull requests plan ordinary task additions directly from the task path and
member record. They run deterministic contracts, prospective digests, and the
exact changed-task Oracle. Merge groups validate the combined inventory.
Scheduled and manual runs own full-portfolio work.

Network policy remains independent from image selection. Task TOML uses
Harbor's native `public`, `no-network`, and `allowlist` values; optional proxy
operation belongs to job composition.

## Consequences

Two unrelated task additions have disjoint intended tracked paths and can merge
in either order. A new task changes the prospective suite digest without
rewriting an existing snapshot lock or task-owned analysis record.

Shared verifier-image migration requires an operator-published immutable image
before copied runtime support can be removed without breaking Oracle
runnability. Until that digest exists, the current runtime is explicitly bound
by profile digest and remains a known migration obligation; unpublished image
references are forbidden.

This ADR supersedes ADR 0008 only for mutable manifest generation, suite-owned
membership metadata, and verifier-support synchronization. ADR 0008 remains
authoritative for direct Harbor task ownership and visibility boundaries.
