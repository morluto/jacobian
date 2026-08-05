# Recover an exact geometric lower-bound certificate

The frozen expression is a sum of two radicals in arbitrary real `a,b`. Select a declared proof method and submit an exact distance-model certificate: both fixed centers as rational coefficients of `sqrt(2)`, the independently expanded radicand coefficient records, the squared distance between the centers, the universal lower bound obtained from the triangle inequality, and an exact equality witness proving sharpness.

Each `scaled_centers` entry is an integer pair `[p, q]` encoding the center coordinate `(p/2) * sqrt(2), (q/2) * sqrt(2)`; the two centers may be listed in either order. The verifier derives each radicand coefficient record from the submitted center and compares the resulting multiset against the frozen expression, so the expanded radicands must be paired with their corresponding centers but need not follow a fixed ordering.

Sampling is not a universal proof. Claim only `COMPUTED`; the verifier checks exact algebra in `Q(sqrt(2))` but does not invoke a proof assistant.

The evidence file at `evidence/radical-distance-certificate.json` must be a JSON object with exactly the keys `schema_version`, `task_id`, `result`, and `limitations`. The `schema_version` must be `"1"`, `task_id` must match the submission, and `result` and `limitations` must be byte-for-byte copies of the corresponding submission fields. The evidence descriptor in `submission.json` must bind this file by its `sha256:` digest, and the evidence file must be no larger than 16 MiB.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SHARP_LOWER_BOUND_CERTIFIED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/radical-distance-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/radical-distance-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
