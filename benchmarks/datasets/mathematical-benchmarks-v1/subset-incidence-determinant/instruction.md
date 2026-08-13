# Factor the nonempty-intersection matrix

For the matrix indexed by nonempty subsets of `[n]`, with entry one exactly
when two subsets intersect, derive its determinant for every `n>=1`.

Submit an exact incidence factorization.  Choose and describe a valid
`sample_n`, provide a cardinality-then-mask ordering of all nonzero bitmasks and
the diagonal weights in that basis.  Also provide the determinant parity trace
for every `n` through `trace_max_n`, including the number of nonempty even-cardinality
subsets and the resulting determinant.

The public determinant values alone are insufficient.  The verifier rebuilds
the intersection matrix and checks the submitted factorization exactly, then
recomputes the parity formula.  Bind `/app/evidence/answer.txt`; do not claim
`VERIFIED`. Scope the claim to all nonempty-subset intersection matrices for
`n>=1`. In `limitations`, state that the finite incidence-factorization replay
does not replay the universal theorem in Lean.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `DETERMINANT_CLASSIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
