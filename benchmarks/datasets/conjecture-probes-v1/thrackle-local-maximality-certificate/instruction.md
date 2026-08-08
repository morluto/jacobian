# Exact local thrackle certificate

On the five frozen rational points, select exactly five of the ten candidate straight-line edges so every pair of selected edges meets exactly once: either at a shared endpoint or in one proper interior crossing.

Submit the sorted selected edges, a complete classification of all ten selected-edge pairs (`SHARED_ENDPOINT` or `PROPER_CROSSING`), and, for each excluded candidate edge, the lexicographically first selected edge disjoint from it. The latter witnesses that no excluded edge can be added while preserving the thrackle property.

The verifier independently evaluates orientation determinants, endpoint incidence, every pair classification, and every local-maximality witness. Collinear overlap and mere line intersection outside the closed segments do not count. Edge and witness order is canonical.

This Regression certificate proves only a five-edge thrackle locally maximal inside this frozen `K5` candidate universe. It does not prove Conway's Thrackle Conjecture. Claim at most `CHECKED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact five-edge thrackle and local-maximality certificate inside one frozen K5 universe.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `LOCAL_THRACKLE_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
