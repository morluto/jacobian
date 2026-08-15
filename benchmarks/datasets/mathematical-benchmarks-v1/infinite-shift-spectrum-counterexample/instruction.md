# Audit the missing finite-dimensional hypothesis

The frozen statement claims that `S*T` and `T*S` have equal eigenvalue sets for arbitrary endomorphisms. Work on the rational vector space with basis `e_0,e_1,...` whose vectors have finite support.

Choose which operator is the right shift `R(e_i)=e_(i+1)` and which is the left shift `L(e_0)=0`, `L(e_(i+1))=e_i`. Report the actions of `S`, `T`, `ST`, and `TS` on every basis vector from 0 through 8. Identify which composition is the identity, which has `e_0` as a zero eigenvector, and the missing finite-dimensional hypothesis. Explain how the symbolic shift rules establish these identities for every basis vector; the finite basis window is a replay check, not an exhaustive model of the infinite-dimensional space.

The verifier reconstructs both shift rules and compositions. Here “eigenvalue set” means the point spectrum, not the full operator spectrum. The finite basis window is a replay check, not an exhaustive model of the infinite-dimensional space.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
