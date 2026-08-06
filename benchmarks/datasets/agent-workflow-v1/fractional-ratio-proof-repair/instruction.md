# Repair a binary fractional-ratio proof

The frozen problem maximizes `(alpha + t·x)/(beta + f·x)` over binary vectors. The public proof instead analyzes a linear benefit objective with a budget constraint and fractional decision variables.

Identify the three contract mismatches exactly. Then solve the frozen 24-item instance and submit an exact residual optimality certificate. If the attained reduced ratio is `p/q`, report every residual `q*t_i - p*f_i`, the affine constant `q*alpha - p*beta`, the indices with positive residual, and the maximum residual sum obtained by independently choosing each binary coordinate. A zero maximum residual proves no binary vector exceeds `p/q`; the submitted vector must attain equality.

Write `/app/submission.json` and bind one text explanation at `/app/evidence/answer.txt`. The independently replayed typed certificate belongs in `submission.json`; no duplicate private serialization is required in the prose. Explain both the proof mismatch and why the residual certificate repairs the frozen objective. Do not claim a general theorem is machine verified.

The evidence file must be at most 65536 bytes and its first line must be exactly: `The public proof replaces the ratio objective, relaxes the binary domain, and adds an undeclared budget. The exact residual certificate repairs the frozen objective: every coordinate is chosen by its signed residual and the maximum transformed residual is zero.` Additional lines are allowed and ignored.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions. The limitations array must exactly state: The verifier certifies only the frozen exact instance; it does not machine-prove a general greedy theorem.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PUBLIC_PROOF_INVALID_REPAIR_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
