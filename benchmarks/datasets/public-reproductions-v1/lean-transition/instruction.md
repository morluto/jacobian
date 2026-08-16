# Lean proof-state tactic application

Apply the declared tactic to the Lean proof state in `input.json` and report the resulting goal count and whether the proof is completed. Write `submission.json` to the exact agent-visible `submission_schema.json`. Claim `COMPUTED` assurance; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier matches the submitted goal count and completion flag against the frozen CORE tactic transition determined by this statement, prefix, and tactic.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
