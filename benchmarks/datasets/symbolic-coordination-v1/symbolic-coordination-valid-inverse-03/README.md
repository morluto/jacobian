# jacobian/symbolic-coordination-valid-inverse-03

Assess one exact polynomial-map claim in the valid-two-sided-inverse pilot family.

## Case

- family: `valid-two-sided-inverse`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@1`
- fixture digest: `sha256:bca5b39f6101cf26f3b5cff541fae35c1f5297db5c05898e83e3a3c2b67f68e9`
- note: Affine translation and shear.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
