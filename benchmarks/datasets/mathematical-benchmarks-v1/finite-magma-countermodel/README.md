# jacobian/finite-magma-countermodel

Find a smallest finite countermodel to an implication between two magma
identities.

## Field

algebra

## Provenance

- case_version: mathematical-benchmarks-v1
- contamination_class: internal-trace-derived-structural-variant
- fixture_digest: sha256:0efab4007c68dc526ace6a4e2e39c9767d7d0f5b3f75a20dff626b2f62ffedcb
- upstream: internal-conversation:postdoc-followup-magma-20260728
- derivation: A frozen two-variable finite-magma implication with exhaustive
  search orders one and two.
- derivation_note: Independent review replays the complete frozen finite
  fixture. The uncommitted source conversation is not treated as proof or
  held-out evidence.

## Contract

- schema_version: 1.4
- difficulty: medium
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
operation, or invocation order. The clean-room verifier independently checks
the complete operation table, every valuation of the premise, the submitted
target refutation, and every smaller declared carrier order. A wrong result or
an unsupported `VERIFIED` claim forces reward to zero.
