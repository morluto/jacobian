# Lean premise retrieval

Retrieve candidate premises/tactics for the Lean statement in `input.json` and report the top candidate as a structured tactic with a command, theorem identifier, and argument identifiers. Also report whether retrieval was exhaustive. Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier matches the submitted tactic against the frozen MATHLIB retrieval record determined by this statement and proof prefix.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
