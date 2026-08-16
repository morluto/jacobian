# jacobian/symbolic-coordination-valid-inverse-01

Assess one exact polynomial-map claim in the valid-two-sided-inverse pilot family.

## Case

- family: `valid-two-sided-inverse`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:053db45b425e1e131361cfede70c06fb26ac416961a2588e3194c01cd9d5f606`
- note: Two-variable triangular shear.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
