# jacobian/symbolic-coordination-near-miss-03

Assess one exact polynomial-map claim in the perturbed-near-miss pilot family.

## Case

- family: `perturbed-near-miss`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:911107a7b5541f60863a602a25f52798564f22f5cae782f51eed5d63ad5d7dd7`
- note: Affine inverse constant perturbed.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
