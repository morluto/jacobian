# jacobian/symbolic-coordination-semantic-equivalence-02

Assess one exact polynomial-map claim in the semantic-equivalence pilot family.

## Case

- family: `semantic-equivalence`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:08e00dc4d5685f0fb8e2c93e5f5396b79aa00aa8fee2a49964777e36cba96ef6`
- note: Affine duplicate terms and renamed source/target variables.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
