# Lean proof-state tactic application

Apply the declared tactic to the Lean proof state in `input.json` and report the resulting goal count and whether the proof is completed. Write `submission.json` to the exact agent-visible `submission_schema.json`. Claim `COMPUTED` assurance; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
