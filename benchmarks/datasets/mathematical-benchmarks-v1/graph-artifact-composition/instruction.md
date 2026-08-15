# Graph artifact composition

Using the labelled graph in `input.json`, determine the complete set of
maximum-degree vertices, the shortest-path distance from every vertex to that
set in lexicographic vertex order, the maximum such distance, and every vertex
attaining it. Preserve the calculation in `evidence/answer.txt`, include that
file's SHA-256 digest, and write `submission.json` to the exact agent-visible
`submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
