# Lean proof-state and trust inspection (issue #95)

Status: design track for durable Lean intermediate inspection contracts.

## Missing outcomes

Durable, independently callable proof-state and trust inspection beyond:

- `lean.statement.propose` (elaboration status),
- `lean.proof_state.apply_tactic` (replay-source tactic steps),
- `lean.check` (complete proof verification).

## Contract decisions

- Environment, source revision, and import digests bound into state identity.
- Durable proof-state identity, expiry, and stale-state rejection on drift.
- Normalized goals, local context, metavariables, and local instances.
- Term-application diagnostics distinct from tactic syntax failures.
- Dependency closure, axioms, package manifests, and sorry/admit reporting.
- Session workers vs replayable immutable state artifacts (prefer immutable
  artifacts for independent inspection).

## Verification boundary

Proof-state transitions and inspections are `COMPUTED` operational evidence.
Only a complete clean replay through operator-pinned `lean.check` may return
`VERIFIED`. Trust inspection reports dependencies; it does not set trust policy.

## Non-goals

- Replacing Lean as the proof kernel.
- Letting inspection APIs authorize `VERIFIED`.
- Prescribing a proof strategy through discovery rankings.

This note does not ship new capabilities; it freezes the acceptance boundary for
implementation after overlapping Lean contract work lands.
