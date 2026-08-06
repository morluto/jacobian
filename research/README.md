# Research records

Research records document capability discovery and its task- or
snapshot-owned handoffs. They are separate from executable Harbor benchmarks:

- `benchmarks/datasets/` contains executable Harbor cases;
- `benchmarks/tooling/` contains reusable Harbor infrastructure; and
- no active suite-wide evaluation bundle is maintained under `research/`.

Records under `research/` are not injected into agent containers and must not
be used as performance evidence without a separately frozen held-out
evaluation. A research record may reference a canonical dataset, but it must
not duplicate task paths, fixtures, Oracle material, or Harbor job inputs.

Evaluation handoffs are kept with the task or immutable snapshot that owns
them. The committed `mathematical-benchmarks-v1` lock under `benchmarks/snapshots/`
defines the reproducible task set. Task-owned `analysis/gap.json` records carry
optional discovery context as explicitly historical provenance; they are not a
mutable suite-wide ledger.
