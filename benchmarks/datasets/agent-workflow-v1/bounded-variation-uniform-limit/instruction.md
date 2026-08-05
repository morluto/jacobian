# Separate uniform convergence from variation convergence

On `[0,2*pi]`, choose an integer `q` with `2 <= q <= 9` and use

`f_n(x) = sin(q*n*x)/(q*n)`, for `n >= 1`.

Submit a certificate that `f_n` converges uniformly to zero while every
`f_n` has total variation exactly four. State the general sup-norm bound and
the exact monotone-segment accounting: two endpoint segments and all interior
segments. Include at least four distinct freely chosen positive indices with
their frequency, amplitude, segment counts, endpoint contribution, interior
contribution, and total variation.
In `result.argument`, use the three typed values
`SUP_NORM_1_OVER_QN_TENDS_TO_ZERO`, `TOTAL_VARIATION_IS_CONSTANTLY_FOUR`,
and `UNIFORM_CONVERGENCE_DOES_NOT_FORCE_VARIATION_CONVERGENCE` to record the
mathematical explanation.

The verifier recomputes every integer and rational identity. Sampling, a graph,
or a conclusion label alone is insufficient. Evidence must contain exactly one
`RESULT_JSON:` line equal to `result`; optional surrounding prose is not a
substitute for the typed `result.argument` values. Set `limitations` to the
single structured value `NO_PROOF_ASSISTANT_VERIFICATION`.
The digest-bound evidence file must not exceed 16 MiB.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `UNIFORM_CONVERGENCE_DOES_NOT_FORCE_VARIATION_CONVERGENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
