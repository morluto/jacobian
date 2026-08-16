# jacobian/symbolic-coordination-search-timeout-01

Assess one exact polynomial-map claim in the bounded-collision-scope pilot family.

## Case

- family: `bounded-collision-scope`
- case version: `symbolic-coordination-v1/pilot-1`
- generator: `symbolic-coordination-pilot-generator@2`
- fixture digest: `sha256:a0d4175f3c0503416fe8fa9f996d6ffb9408da18888156ed350ed6bab2cb276f`
- note: Timeout without a witness is a non-conclusion.

## Verification boundary

The task is offline and solvable without Jacobian. The instruction names no
operation or tool order. The task-local clean-room verifier imports neither
Jacobian nor the generator; it replays exact rational polynomial arithmetic,
input and claim bindings. Reward is binary: the replayed mathematical
predicate and every required binding must hold.
