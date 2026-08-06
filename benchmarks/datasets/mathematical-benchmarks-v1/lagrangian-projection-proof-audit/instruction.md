Audit the frozen projection step from the supplied research-proof correction.

Submit an exact rational full-rank matrix `D` for which `L = D^T J D` is nonzero, together with nonzero coefficient matrices `P` and `Q`. Reconstruct `W = D P + J D N Q`, where `N = (D^T D)^{-1}`. Report the Gram matrix, its inverse, the Lagrangian defect, both naive projections, and both corrected projection expressions.

Your witness must make each naive projection differ from its intended coefficient while both corrected identities hold exactly. Rational strings must be canonical. Bind one evidence file and do not claim `VERIFIED`; the paper's main theorem and the corrected analytic proof remain outside scope. Include one `RESULT_JSON:` line in `evidence/answer.txt` containing the exact submitted `result` object as compact JSON, so the explanation is bound to the certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `HIDDEN_LAGRANGIAN_ASSUMPTION_REFUTED_AND_PROJECTIONS_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
