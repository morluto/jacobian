# jacobian/jcb-postdoc-004

Research diagnostic: Written on the Wall II Conjecture 194: discover an 18-vertex obstruction

## Field

extremal-graph-theory

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:cc04ceff88324e46a21e5a9acf63d2d2803b42638277a4f56ac8a9c7ed286b04
- derivation: Unlike the supplied-witness closure cases, this task asks the agent to discover a compact structured counterexample while using exact graph invariants to guide and check the search.

## Portfolio status

- historical_fit: `PARTIAL`
- current_status: `PARTIAL`
- evaluation_status: `RUNNABLE_PUBLIC_REPRODUCTION`
- next_action: Run the unchanged public prompt to inspect composition and discovery; do not add an opaque counterexample-search workflow.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The v2 contract requires a canonical
graph certificate rather than an answer summary. Its clean-room verifier
recomputes connectivity, global and neighborhood independence numbers, the
exact rational inequality, and Hamiltonian-path nonexistence. It accepts any
valid graph with at most 20 vertices; wrong results and unsupported VERIFIED
claims force reward to zero.
