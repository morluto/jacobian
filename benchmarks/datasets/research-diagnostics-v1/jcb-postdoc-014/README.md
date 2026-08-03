# jacobian/jcb-postdoc-014

Research diagnostic: Nine-line counterexample to combinatorial determination of Jacobian-relation degree

## Field

algebraic-geometry

## Provenance

- case_version: research-diagnostics-v1
- contamination_class: public-answer-visible-diagnostic
- fixture_digest: sha256:f664651eacddbad9976b2cce12f651d4651323c6f5a0976c6c690868d7f5dde8
- derivation: Two rational arrangements of nine projective lines have isomorphic intersection lattices but different minimal degrees of Jacobian relations. The decisive computation is an exact graded-kernel calculation, close to Jacobian's polynomial and rational-linear portfolio but not currently exposed as one domain-owned operation.

## Portfolio status

- historical_fit: `MISSING`
- current_status: `COVERED`
- evaluation_status: `REGRESSION_COVERED`
- next_action: Run repeated public model reproductions under the frozen no-retrieval profile; the v1 MISSING label remains historical and must not be edited.

## Contract

- schema_version: 1.4
- difficulty: hard
- maximum_assurance: COMPUTED
- agent-visible verification record schema: no
- timeout_sec (agent): 600.0
- timeout_sec (verifier): 120.0
- environment_mode: separate

The task is self-contained and offline. The v2 contract requires exact sparse
syzygies rather than a public-answer summary. Its clean-room verifier rebuilds
the arrangements, checks their projective flats, proves lower-degree
injectivity by exact modular rank, and replays the submitted polynomial
relations over the integers. Wrong results and unsupported VERIFIED claims
force reward to zero.
