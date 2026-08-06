# Audit shallow Lean axiom reports

The frozen input contains two declaration-dependency graphs and the axiom sets
reported by the affected Lean collector. For each case, reconstruct the full
transitive dependency closure from the declared roots, compare it with the
observed report, and list the missing dependencies in sorted order.

Classify each report as `COMPLETE` or `INCOMPLETE`. Classify every missing
dependency by its frozen role. Bind a concise explanation at
`/app/evidence/answer.txt` and write `/app/submission.json`.

Do not claim proposition truth, current Lean behavior, or independent
reproduction of the upstream issue. The checker audits only the frozen graphs
and permits at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `SHALLOW_REPORTS_INCOMPLETE`, `INCOMPLETE_AUDIT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
