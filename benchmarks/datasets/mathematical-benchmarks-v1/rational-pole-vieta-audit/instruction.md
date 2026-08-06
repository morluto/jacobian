# Audit a rational-equation root-sum proof

The frozen trace claims that the poles of
`sum_{k=1}^4 k/(x^2-k) = 2010x-4` are `1,2,3,4`. Diagnose that step and repair
the computation of the sum of all complex solutions.

Submit `/app/submission.json` following `/app/submission_schema.json` and
digest-bound prose at `/app/evidence/answer.txt`. Coefficient arrays are
low-to-high. Provide the common denominator, combined numerator, cleared
polynomial, the value of the surviving numerator at each denominator square
value `k=1,2,3,4`, and the resulting root sum. The verifier reconstructs the
rational equation and checks that clearing denominators introduced no pole
roots.

Do not claim proof-assistant verification. Claim `COMPUTED` assurance and
complete scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently reconstructs and checks the exact polynomial and pole-domain certificate.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FLAGGED_POLE_STEP_INVALID_REPAIRED_SUM`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
