# Audit disjoint closed sets versus positive distance

The frozen ProofNetVerif pair strengthens “separated” to a uniform positive-distance claim. Construct two disjoint closed subsets of the Euclidean plane that refute that strengthening.

Use a parametric pair

- `A_n = (h*n, 0)`;
- `B_n = (h*n, s/(n+c))`, for positive integer `n`;

where `2 <= h <= 20`, `1 <= s <= 20`, and `2 <= c <= 20`. Submit ten distinct sample indices and their exact squared same-index distances. For each epsilon `1/k`, `k=2,...,9`, submit an index whose distance is strictly smaller.

Also submit the exact lower bound on squared horizontal separation between distinct indices. This is the certificate that each family has no finite accumulation point and is closed. State the formalized positive-distance conclusion and the corrected conclusion separately.

Write `/app/submission.json` and the digest-bound evidence file required by the schema. Claim only `COMPUTED`; this benchmark does not run Lean or machine-prove general topology.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact parametric metric replay; Lean elaboration and foundational topology remain outside scope.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `DISJOINT_CLOSED_DOES_NOT_IMPLY_POSITIVE_DISTANCE`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/disjoint-closed-distance-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/disjoint-closed-distance-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
