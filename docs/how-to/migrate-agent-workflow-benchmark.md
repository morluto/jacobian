# Migrate the former agent-workflow dataset

The active dataset identity `jacobian/agent-workflow-v1` was retired and
replaced by `jacobian/mathematical-benchmarks-v1`. The rename describes the
corpus more accurately: it is a fixed mathematical benchmark contract, while
workflow observation is one permitted use of its jobs.

Task IDs do not change. Five answer-visible or publicly derived reproductions
now belong to `public-reproductions-v1`:

- `balanced-row-permutation`
- `closed-set-distance-strengthening-audit`
- `coin-process-potential`
- `cyclic-vector-inequality`
- `superposition-proof-replay`

Consumers must select the new dataset ID explicitly. There is no active alias
for the retired name, so stale job or configuration references fail closed.
Update dataset paths, registry IDs, `case_version`, and
`evaluation_owner` together. The historical lock under
`benchmarks/snapshots/agent-workflow-v1/` and ignored result directories are
evidence of the old evaluation boundary and must remain byte-for-byte intact;
create a new immutable lock for every intentional publication of the renamed
or expanded dataset.

Subject organization is metadata, not filesystem layout. Each member carries a
controlled `primary_domain` and a detailed `field`, and task bundles remain
direct children of their dataset root.
