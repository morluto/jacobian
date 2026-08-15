# jacobian/symbolic-coordination-valid-inverse-02

Assess one exact polynomial-map claim in the valid-two-sided-inverse pilot family.

## Case

- family: `valid-two-sided-inverse`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@1`
- fixture digest: `sha256:94e307fab75fd385476d04490d2552da100ead22e0e85b4f14d27dca2977429b`
- note: Triangular shear in the second coordinate.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
