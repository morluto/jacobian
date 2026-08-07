# Construct a nontrivial finite Hadwiger certificate

Submit a simple connected triangle-free graph on vertices `0..10` with exactly
20 edges and minimum degree at least 3. Supply a proper four-coloring and four
pairwise-disjoint nonempty branch sets witnessing a `K4` minor: every branch
set must induce a connected subgraph and every pair must have a crossing edge.

The verifier independently rejects every possible three-coloring by exact
backtracking, checks the submitted four-coloring, and validates the complete
minor model. Complete graphs, triangles, isolated padding, and label-only
chromatic claims are rejected.

Evidence is matching JSON with exactly `schema_version`, `task_id`, `result`,
and `limitations`: `schema_version` must be the string `"1"`, `task_id` must
equal the task id, `result` must exactly copy the submitted `result` object
(including JSON types), and `limitations` must equal the declared limitations.
This checks one finite graph only and does not prove Hadwiger's conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact finite coloring and minor replay only; no global Hadwiger conclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FINITE_HADWIGER_K4_CERTIFICATE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
