# jacobian/symbolic-coordination-valid-inverse-04

Assess one exact polynomial-map claim in the valid-two-sided-inverse pilot family.

## Case

- family: `valid-two-sided-inverse`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:7e842ccd456a5d059b344dd0961062b6719a837a97098a40704f8d9513645a1c`
- note: Linear inverse with rational coefficients.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
