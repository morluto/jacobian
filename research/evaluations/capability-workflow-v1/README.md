# Capability workflow evaluation v1

This directory contains the non-runnable discovery handoff and comparison
design derived from the public Harbor `agent-workflow-v1` suite. It is
research metadata, not a Harbor dataset: it contains no executable tasks, job
configuration, Compose file, model result, or held-out score.

The executable public cases remain task bundles directly under the
`benchmarks/datasets/agent-workflow-v1` Harbor dataset. Reusable Harbor support lives
under `benchmarks/tooling`; this directory may reference those datasets and
tools, but it does not copy their task lists or inject research records into
agent containers.

## Snapshot-scoped, leaf-owned records

The gap analysis is migrated to leaf-owned, snapshot-scoped records so that the
historical ledger is frozen to one immutable benchmark snapshot and is not
retargeted when the live suite gains unrelated tasks.

- Each public task bundle owns its analysis record at
  `benchmarks/datasets/agent-workflow-v1/<task-id>/analysis/gap.json`. The leaf
  carries the per-task mathematical moves, current-capability boundaries, gap
  classes, contamination, candidate ids, disposition, and next action. It is
  snapshot-independent: it carries no `snapshot` block, because embedding a
  `snapshot_id` would create a digest cycle (the leaf is inside the Harbor task
  checksum). It is research metadata sitting alongside the bundle; it is not a
  Harbor task artifact and is not visible to evaluated agents.
- `gap-ledger.json` is the generated historical ledger. It is the only place
  that binds the immutable snapshot (the `snapshot` block) and a frozen,
  ordered `analysis_records` list. Each entry pins `order`, `task_id`, the leaf
  `analysis_ref`, and the canonical-JSON `analysis_digest` of that leaf's
  `analysis` content. The records are inline frozen data: they are not derived
  from current suite membership, a mutable publication manifest digest, or a
  hard-coded current task count. Adding a new task to the live suite after
  capture does not rewrite this ledger.
- `comparison-plan.json` references the same `snapshot` block as the ledger for
  its public reproduction fixture, instead of a mutable manifest digest and
  task count.

## Immutable benchmark snapshot

The `snapshot` block in both `gap-ledger.json` and `comparison-plan.json`
references the committed content-addressed lock under
`benchmarks/snapshots/agent-workflow-v1/`. That lock freezes the ordered task
set, Harbor task digests, environment profiles, verifier runtime, Harbor
version, source tree, and evaluation configuration. Later task additions do
not rewrite the lock or these historical records. Publication tooling renders
Harbor's `dataset.toml` beneath ignored `dist/harbor/` from the lock.

The leaf `analysis/gap.json` files carry
no snapshot binding by design, so they need no update when the lock is
generated. Until then the records are frozen by their leaf content digests and
order, which is sufficient to keep the historical analysis stable; the
snapshot identity is the one remaining integration value.

## Artifact roles

- `gap-ledger.json` classifies the mathematical moves and portfolio gaps
  exposed by the frozen public task set. Its accepted candidates are discovery
  handoffs, not performance claims. `runtime_snapshot` preserves the captured
  handoff/runtime facts (repository tree, package and Harbor versions, catalog
  and policy digests, provider availability, and historical Oracle coverage).
- `comparison-plan.json` fixes the C0/C1/C2 meanings, matched-condition
  invariants, contamination boundary, and execution gate. Its conditions have
  no committed Harbor job while that gate is closed.

## Historical Oracle coverage

`runtime_snapshot.oracle_evidence` records the Oracle coverage captured at
capture time (a subset of the snapshot task set). It is historical coverage,
not the snapshot identity and not the full frozen `analysis_records`
membership. It must not be confused with the snapshot lock's ordered task
digests.

## Evidence limits

The public reproduction is a fixed, answer-visible suite and must never be
cited as causal evidence. C1 is the frozen current catalog; C2 is the same
catalog plus exactly one candidate delta. A future held-out run must supply a
separately frozen task set whose Oracle material is stored outside the
evaluated workspace, use at least two independent mathematical families, and
bind every result to the exact catalog, policy, image, model, prompt, budget,
seed, scorer, and task digests.

No held-out fixtures or Oracle answers are committed here. Until those
external identities are filled and a hard model-run budget is authorized,
`comparison-plan.json` remains `SCAFFOLD_ONLY` and no model execution is
allowed. Plans and handoffs in `research/evaluations/` are not performance
evidence without a separately frozen held-out evaluation.
