# SAT decision with witness

Decide the exact CNF formula in `input.json`. For a satisfiable formula, return `SATISFIABLE` and a complete Boolean assignment satisfying every clause. For an unsatisfiable formula, return `UNSATISFIABLE`. Record the clause-by-clause check in `evidence/answer.txt`, include its SHA-256 digest, and write `submission.json` to the exact agent-visible `submission_schema.json`. Claim `COMPUTED` assurance; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
