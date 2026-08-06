# Audit a locally wrong inversion definition masked by a correct aggregate

The frozen formalization calls a pair `(i,j)` an inversion when `sigma[i] <= sigma[j]`; the intended definition uses `sigma[i] > sigma[j]`. For `n=4`, produce `/app/submission.json` following `/app/submission_schema.json` and `/app/evidence/inversion-audit.json` following `/app/evidence_schema.json`.

Supply any permutation witnessing different pointwise counts. Also report the independently computed sums of both counts over all 24 permutations and explain, through the typed complement relation, why the wrong definition nevertheless satisfies the published aggregate formula. The witness is not fixed; valid alternatives are accepted. Claim only `COMPUTED`. The verifier exhaustively recomputes both functions and both aggregates.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `LOCAL_DEFINITION_WRONG_AGGREGATE_MASKS_DEFECT`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/inversion-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/inversion-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
