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

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
