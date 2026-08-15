# Finite partition and exact coverage

Partition the exact universe in `input.json` into named cases according to the
given residue relation. Return every member exactly once. State `TRUE` only
when the cases are pairwise disjoint and cover the complete supplied universe.
Write `submission.json` to the exact schema in the agent-visible
`submission_schema.json`. Put the coverage calculation in `evidence/answer.txt`
and include that file's SHA-256 digest in the evidence list.
<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
