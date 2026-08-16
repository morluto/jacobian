# jacobian/symbolic-coordination-semantic-equivalence-04

Assess one exact polynomial-map claim in the semantic-equivalence pilot family.

## Case

- family: `semantic-equivalence`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:4198243c40946bb870d3dec163482fad2a7b1e1b52a59c3e23e077e58e79a2d5`
- note: Rational coefficients plus cancelling sparse terms and renaming.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
