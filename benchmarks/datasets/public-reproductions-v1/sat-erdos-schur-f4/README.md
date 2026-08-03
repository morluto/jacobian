# jacobian/sat-erdos-schur-f4

Determine the Erdos-Schur number f(4), submit an explicit sum-free partition
for the lower bound, and rely on the independent verifier-owned CSP replay for
the upper bound.

## Field

logic

## Provenance

- case_version: public-reproductions-v1
- contamination_class: public-answer-visible-reproduction
- fixture_digest: sha256:07fcfc8737ac90932553976a9d490205d1095f7f7cd922118e9b97ea9d79b188
- derivation: Erdos-Schur f(4)=45 agent regression; answer-visible diagnostic.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no agent tool,
capability, or invocation order. The verifier reconstructs and exhausts the
finite coloring CSP with propagation and color-symmetry reduction, while the
submitted partition is checked directly. It scores correctness, evidence validity, scope
accuracy, assurance calibration, and aggregate reward; a wrong result or an
unsupported VERIFIED claim forces the reward to zero.
