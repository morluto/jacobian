# jacobian/symbolic-coordination-near-miss-01

Assess one exact polynomial-map claim in the perturbed-near-miss pilot family.

## Case

- family: `perturbed-near-miss`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:925f7ca8e00ccba6c77e69896eb376de68286ca264f220f8349c5b6a4cfc226e`
- note: Quadratic inverse coefficient perturbed.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
