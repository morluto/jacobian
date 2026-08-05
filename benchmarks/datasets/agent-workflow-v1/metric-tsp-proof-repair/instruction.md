# Repair a Metric TSP proof

Audit and repair the flawed claim in the offline input. Identify the invalid
inference and state the strongest generic guarantee supported by the
double-tree argument. For the supplied metric, give a minimum spanning tree,
an Euler circuit of its doubled edges, the Hamiltonian cycle obtained by
retaining first visits, and an optimal Hamiltonian cycle. Report all exact
weights so the concrete trace demonstrates why exactness is unsupported while
the repaired guarantee holds.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a concise proof repair and weight calculation in `evidence/answer.txt`,
include a `RESULT_JSON:` line containing the submitted result as JSON, and bind
the file with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
