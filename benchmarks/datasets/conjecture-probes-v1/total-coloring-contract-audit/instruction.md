# Total-coloring contract audit

The frozen validator checks a proper vertex coloring and a proper edge coloring separately, then incorrectly calls their union a total coloring. Audit that contract on the supplied Petersen graph.

Submit `/app/submission.json` and one bound `/app/evidence/answer.json`. Your certificate must contain:

1. a **flawed-pass assignment** using exactly the declared four-color palette whose vertex projection is proper and whose edge projection is proper, but which has at least one vertex–incident-edge color collision;
2. the complete collision list for that assignment (the order is not significant); and
3. a **repaired assignment** that is a valid total coloring of all ten vertices and fifteen indexed edges.

Both assignments must list ten vertex colors in vertex order and fifteen edge colors in the frozen edge order. The verifier independently reconstructs every vertex adjacency, edge adjacency, and incidence constraint. It accepts alternative assignments and does not trust submitted collision labels.

The conclusion is limited to this one finite validator defect and its Petersen-graph repair. It neither proves nor disproves the Total Coloring Conjecture. Claim at most `CHECKED`; truth beyond the frozen finite instance is `NOT_ASSESSED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact audit and repair of one projected total-coloring validator on the Petersen graph only.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PROJECTED_VALIDATOR_IS_UNSOUND_AND_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
