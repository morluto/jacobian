# jacobian/symbolic-coordination-semantic-equivalence-01

Assess one exact polynomial-map claim in the semantic-equivalence pilot family.

## Case

- family: `semantic-equivalence`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@1`
- fixture digest: `sha256:842e7930e19e1cd8bb249ad72e175a357efb2d2cb05eda5f6b32ace02cc7174d`
- note: Term reordering, duplicates, cancellations, and variable renaming.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
