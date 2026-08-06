# Gaussian-moment generality audit

The frozen packet gives a three-real-Gaussian template and a formal Lagrange-inversion identity. A prior audit checked twelve moments and then claimed the identities for every exponent. That extrapolation is invalid.

Choose any nonzero rational parameter `a` within the declared bounds. Derive the quadratic `v(z)` and rational inverse branch `zeta(t)` satisfying the identities below; both are determined by `a` and the correction-factor identity rather than freely chosen. Submit canonical rational coefficient lists in ascending degree order.

Your certificate must establish, by exact rational-function identities rather than samples, all of the following:

1. `zeta = t h(zeta)` for `h(z)=a+z`;
2. `1 - 2 t v(zeta) = (1-t)^(-2)`;
3. the constant-term-one square-root branch cancels `1-t h'(zeta)`;
4. `E(exp(tP))=1` and `E(Q exp(tP))=t/(1-t)`;
5. therefore `E(P^m)=0` and `E(QP^m)=m!` for every `m>=1`.

Also classify the twelve-moment argument as insufficient. Do not claim that finite checking proves an all-exponents statement. Do not claim machine verification, settle the two-dimensional case, or assess unrelated reductions in the source paper. State limitations covering both: (1) the certificate is not proof-assistant or machine verification, and (2) it does not settle the two-dimensional case or unrelated claims in the source paper.

Write `/app/submission.json` matching the supplied schema and bind one concise explanation at `/app/evidence/answer.txt`. Include one `RESULT_JSON:` line containing the exact submitted `result` object as compact JSON, so the explanation is bound to the certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `ALL_EXPONENTS_IDENTITY_CERTIFIED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
