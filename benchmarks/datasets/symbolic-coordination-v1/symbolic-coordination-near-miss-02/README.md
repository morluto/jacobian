# jacobian/symbolic-coordination-near-miss-02

Assess one exact polynomial-map claim in the perturbed-near-miss pilot family.

## Case

- family: `perturbed-near-miss`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@1`
- fixture digest: `sha256:0e7a24f9df5665526ba4fe677ca31f295ef103a8403aa110a28c7fcbd0e6fe63`
- note: Inverse shear sign perturbed.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
