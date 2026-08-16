# jacobian/symbolic-coordination-keller-only-01

Assess one exact polynomial-map claim in the constant-nonzero-jacobian pilot family.

## Case

- family: `constant-nonzero-jacobian`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:1f7f99599472304b07b1a7a896a5c3f32b41112a8bb12ec35e7b5a27b0c8e258`
- note: Unit Jacobian triangular map; certificate scope remains Keller-only.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
