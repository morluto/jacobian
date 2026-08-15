# Audit product convergence from marginal convergence

The frozen source claims that `X_n -> X` and `Y_n -> Y` in distribution, together with independence of each pair `(X_n,Y_n)`, imply `X_n Y_n -> XY` in distribution. Audit that inference under the literal assumptions: no joint convergence and no independence of the named limit pair `(X,Y)` are given.

Use the frozen four-point support and marginal law. Model every prelimit pair by one constant-in-`n` independent joint law. Construct a different joint law for `(X,Y)` with exactly the same two marginals. Submit both complete 4-by-4 joint tables and their exact pushforward distributions under multiplication. Identify a product value whose masses differ.

All probability masses are `{numerator, denominator}` objects in `[0,1]`. Equivalent encodings such as `2/200` and `1/100` are accepted after exact `Fraction` normalization. Joint-table entries must be in lexicographic `(x,y)` order and product distributions in ascending product-value order. The verifier independently checks normalization, marginals, prelimit independence, non-product dependence of the limit coupling, and both product pushforwards. Any coupling satisfying the contract is accepted.

Write `/app/submission.json`. Product-distribution entries must be ascending and
may include zero-mass attainable values, which the verifier normalizes away.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
