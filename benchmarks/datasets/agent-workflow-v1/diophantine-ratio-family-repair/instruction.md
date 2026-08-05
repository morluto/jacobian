# Repair an infinite Diophantine-ratio construction

Audit the frozen Vieta-jumping proof for positive integer pairs satisfying `x^2-xy+y^2 | xy(xy-1)`. Identify the first invalid integrality step, then replace the broken recurrence with one integer-polynomial parameter family proving infinitely many distinct ratios.

The `source_audit` object must use the following exact tokens: `invalid_step` is `"VIETA_PARTNER_INTEGRALITY"`, `k` is `"d^2-1"`, `claimed_partner` is `"d^2/(d^2-1)"`, `status_for_d_ge_2` is `"NONINTEGER"`, and `downstream_recurrence_status` is `"UNSUPPORTED"`.

Submit ascending coefficient arrays in the parameter `t` for the reduced variables `a,b,d`, the original pair `x,y`, the norm, both exact identity quotients, the final divisibility quotient, and the ratio polynomial. The verifier binds `x` to `ratio*y` as a polynomial identity, so any valid integer-polynomial parameterization is allowed; do not assume the Oracle's `t^2` parameterization is required.

The verifier independently composes and checks the following polynomial identities over `Z[t]`: `x = d*a`, `y = d*b`, `norm = a^2 - a*b + b^2`, `d^2 - (1-a) = square_congruence_factor * norm`, `d^2*a - 1 = quotient * norm`, `x*y*(x*y - 1) = divisibility_quotient * (x^2 - x*y + y^2)`, and `x = ratio * y`. Choose a sign convention for `a`, `b`, and `d` that satisfies all of these identities simultaneously; the sign-equivalent decomposition `a = -t^2`, `b = -1`, `d = t - t^3` preserves `x`, `y`, and the ratio but does not satisfy the auxiliary congruence identities and will be rejected.

To certify the infinite domain structurally, the verifier expands `x`, `y`, and `ratio` in `s=t-2` and requires nonnegative coefficients, a positive constant term for `x` and `y`, and a positive coefficient of positive degree for `ratio`. This proves positive pairs and a strictly increasing ratio for every integer `t>=2`, rather than relying only on selected probes. Include at least three freely chosen distinct integer probes with `2<=t<=50`; the verifier will independently evaluate the polynomials and divisibility.

Write `/app/submission.json` using the supplied schema and bind `/app/evidence/answer.txt`. The evidence file must contain exactly one line beginning `RESULT_JSON:` whose JSON object equals the submitted `result`; the digest in the submission must match the file. Use scope `the submitted integer polynomial family for t>=2`, completeness `COMPLETE`, the declared limitation, and `COMPUTED` assurance. Do not claim a classification of all solutions or `VERIFIED` assurance.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `FROZEN_PROOF_REPAIRED`, `FROZEN_PROOF_VALID`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
