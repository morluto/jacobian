# Audit a prime-to-prime-power inference

The frozen proof partitions a sum into `m/p` cycle sums for every prime `p | m`. It establishes
that each cycle sum is divisible by `p`, then concludes that the total is divisible by `m`.

Submit a compressed countermodel to that inference. Choose a prime `p` and exponent `e >= 2`, set
the compatibility field `r = 1`, and use `m = p^e <= 10000` with at least four cycles. Give two to
six distinct cycle-sum groups. Each group records a positive multiplicity and a positive cycle sum;
the multiplicities must total `m/p`, every cycle sum must be divisible by `p`, and at least two
different cycle sums must occur. The resulting total must be divisible by `p` but not by `m`.

Report the exact p-adic valuations of `m` and of the total, the local/global statuses, and the
missing proof obligation. Write `/app/submission.json` and one digest-bound JSON evidence file at
`evidence/divisibility-audit.json`. The evidence object must contain exactly
`schema_version: "1"`, the task ID, the result, and the limitations, with the latter three values
matching the submission. Maximum assurance is `COMPUTED`; this audit invalidates one proof step and
does not disprove the source theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier reports protocol dimensions separately. The JSON evidence object must contain exactly schema_version (the string "1"), task_id, result, and limitations; result and limitations must exactly repeat the submission with JSON types preserved. JSON numbers with integral values are valid wherever the submission schema declares type integer; booleans and non-integral numbers are not.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PRIMEWISE_TO_MODULUS_INFERENCE_INVALID`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/divisibility-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/divisibility-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
