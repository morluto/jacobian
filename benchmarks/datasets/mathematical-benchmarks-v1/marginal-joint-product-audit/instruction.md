# Audit product convergence from marginal convergence

The frozen source claims that `X_n -> X` and `Y_n -> Y` in distribution, together with independence of each pair `(X_n,Y_n)`, imply `X_n Y_n -> XY` in distribution. Audit that inference under the literal assumptions: no joint convergence and no independence of the named limit pair `(X,Y)` are given.

Use the frozen four-point support and marginal law. Model every prelimit pair by one constant-in-`n` independent joint law. Construct a different joint law for `(X,Y)` with exactly the same two marginals. Submit both complete 4-by-4 joint tables and their exact pushforward distributions under multiplication. Identify a product value whose masses differ.

All probability masses must be canonical nonnegative rational strings. Joint-table entries must be in lexicographic `(x,y)` order and product distributions in ascending product-value order. The verifier independently checks normalization, marginals, prelimit independence, non-product dependence of the limit coupling, and both product pushforwards. Any coupling satisfying the contract is accepted.

Write `/app/submission.json` and bind a concise explanation at `/app/evidence/answer.txt`. The independently replayed finite laws belong in the typed result; no duplicate private serialization is required in the prose. Explain why marginal convergence does not determine joint convergence or the product law. Product-distribution entries must be ascending and may include zero-mass attainable values, which the verifier normalizes away. Do not claim that a general probability theorem or the original prose has been machine verified.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions. The evidence explanation must state why marginal convergence does not determine joint convergence or the product law; unrelated text does not earn evidence credit. Product distributions may include ascending zero-mass attainable entries, which the verifier normalizes away. In the solver's own words, the limitations array must disclose that this exact finite-law countermodel does not prove a general weak-convergence theorem or disambiguate the original prose.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `MARGINAL_CONVERGENCE_INSUFFICIENT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
