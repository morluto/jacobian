# Multi-tool coordination certificate

Replay y0, c00_y, objective, and the derived scalar m00=y0+c00_y. Certify the declared matrix positive definite with all exact positive leading principal determinants. The scalar m00 is not a matrix entry.

Return the terminal mathematical object described by `submission_schema.json`.
Use exact arithmetic and any mathematical method. Choose your own decomposition,
representations, tools, verification timing, and stopping rule; no tool sequence is
prescribed. Write the mirrored JSON certificate to `evidence/certificate.json`.
Do not widen the declared finite scope or claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Write evidence/certificate.json as a JSON wrapper with exactly schema_version, task_id, result, scope, completeness, and limitations. The last four values must exactly match submission.json. Bind the regular file by SHA-256. The clean-room verifier independently checks the terminal mathematics, frozen input, scope, completeness, evidence, and assurance; prose phrases are not proof evidence.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `LOCAL_SLICE_CERTIFIED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
