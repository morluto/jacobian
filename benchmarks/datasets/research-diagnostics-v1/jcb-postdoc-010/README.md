# jacobian/jcb-postdoc-010

Research diagnostic: Bandeira Matrix AM-GM Open Problem 0.2(a)

## Field

matrix-analysis

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:673101bb112e66e52a86bb0a0b0e058f7068fb2ab23de7cd8156e2c3e2877a17
- derivation: The conjectured spectral-norm inequality was refuted only through a large rational certificate. It is a strong test of exact PSD semantics, compressed witnesses, and independently checkable optimization duality.

## Portfolio status

- historical_fit: `MISSING`
- current_status: `OPEN_GAP`
- evaluation_status: `BLOCKED_ON_INTERVENTION`
- next_action: Keep the exact public certificate as source evidence and define the semantic equivalence and checker obligations before implementation.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The instruction names no tool,
operation, or invocation order. The verifier is a separate clean-room Python
script that scores correctness, evidence validity, scope accuracy, assurance
calibration, and aggregate reward; a wrong result or an unsupported VERIFIED
claim forces the reward to zero.
