# jacobian/symbolic-coordination-near-miss-04

Assess one exact polynomial-map claim in the perturbed-near-miss pilot family.

## Case

- family: `perturbed-near-miss`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@1`
- fixture digest: `sha256:8a61b5bda6610b8d4fe0044103652064c4b41193a1e71bb6992164a2a466f54c`
- note: Linear inverse denominator perturbed.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
