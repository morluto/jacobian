# Audit rational points as a scheme invariant

The frozen source uses an empty scheme to show that `k`-rational points do not determine a `k`-scheme. Produce a stronger nonempty certificate over `k=F_5`.

Let `B=F_5^3` with orthogonal idempotent basis `(f0,f1,f2)`. Let
`A=F_5[u]/(u^3) × F_5 × F_5` with basis `(e0,e1,e2,u,u2)`, where the `ei` are orthogonal idempotents, `e0*u=u`, `e0*u2=u2`, `u*u=u2`, and all other products involving `u` or `u2` vanish. Units are `e0+e1+e2` and `f0+f1+f2`.

Submit both full multiplication tensors, both unit vectors, the coordinate matrix of the unital map `B -> A` sending `fi` to `ei`, and **all** unital algebra maps from each algebra to `F_5` as image vectors on the ordered bases. Give the induced map on rational points by precomposition. Finally exhibit a nonzero nilpotent of exact order three in `A`, while certifying that `B` has no nonzero nilpotent.

The verifier independently enumerates every possible linear functional (`5^5` for `A`, `5^3` for `B`) and checks multiplicativity on all basis pairs. It also rebuilds every product and power. A bijection on the three nonempty rational-point sets therefore coexists with a reducedness obstruction to algebra—and hence affine-scheme—isomorphism.

The submitted tensors, point map, and nilpotent are the executable certificate; no prose explanation or duplicate artifact is required.

The public submission contract is generated below from `tests/public_contract.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
