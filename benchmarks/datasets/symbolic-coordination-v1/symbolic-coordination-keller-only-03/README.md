# jacobian/symbolic-coordination-keller-only-03

Assess one exact polynomial-map claim in the constant-nonzero-jacobian pilot family.

## Case

- family: `constant-nonzero-jacobian`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:d548aedffe66385a3d83bba527db35a31ef6e95a921607c3308603b0fd4011ad`
- note: Constant Jacobian minus one for a linear map.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
