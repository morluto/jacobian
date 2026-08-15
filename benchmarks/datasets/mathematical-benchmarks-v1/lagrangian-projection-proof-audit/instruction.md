Audit the frozen projection step from the supplied research-proof correction.

Submit an exact rational full-rank matrix `D` for which `L = D^T J D` is nonzero, together with nonzero coefficient matrices `P` and `Q`. Reconstruct `W = D P + J D N Q`, where `N = (D^T D)^{-1}`. Report the Gram matrix, its inverse, the Lagrangian defect, both naive projections, and both corrected projection expressions.

Your witness must make each naive projection differ from its intended coefficient while both corrected identities hold exactly. Rational strings must be canonical. Bind one evidence file and do not claim `VERIFIED`; the paper's main theorem and the corrected analytic proof remain outside scope. Include one `RESULT_JSON:` line in `evidence/answer.txt` containing the exact submitted `result` object as compact JSON, so the explanation is bound to the certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
