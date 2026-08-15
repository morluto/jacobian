# Audit a strengthened metric conclusion

The frozen natural theorem says that disjoint closed subsets of a metric space are separated. A
candidate Lean statement instead asserts that there is one positive `epsilon` bounding every
cross-distance from below.

Diagnose this semantic strengthening by constructing two disjoint locally finite subsets of the
rational plane whose distance infimum is zero. Choose a start index from 4 through 20 and submit
eight consecutive indexed point pairs. For every row use `A_n = (n,0)` and `B_n = (n,1/n)`, with
canonical rational coordinate and distance strings. Also submit four to eight distinct positive
canonical rational epsilons in strictly decreasing order, each paired with an index `N` such that
`N` is at least the start index and `1/N < epsilon`; indices must strictly increase.

Identify the natural and predicted conclusions, their semantic relationship, the missing assumption,
and the local-finiteness rule that makes both infinite sets closed. Write `/app/submission.json` and
one digest-bound JSON evidence file at `evidence/distance-audit.json`. The evidence file must be a
JSON object with exactly the fields `schema_version` (the string `"1"`), `task_id`, `result`, and
`limitations`, repeating the submission's task ID, result, and limitations. Include this exact
limitation in the `limitations` array: "The verifier replays exact rational instances and trusts the
standard theorem that locally finite Euclidean subsets are closed; it does not machine-prove the
universal topological argument." Maximum assurance is `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/distance-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
