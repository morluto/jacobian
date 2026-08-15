# jacobian/symbolic-coordination-keller-only-04

Assess one exact polynomial-map claim in the constant-nonzero-jacobian pilot family.

## Case

- family: `constant-nonzero-jacobian`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@1`
- fixture digest: `sha256:158db6bb2b3517a50158320829f707e8912cc7309513ade63c6fd163a58f5672`
- note: Three-variable unit Jacobian triangular map.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
