# Operation discovery regression

Given the natural-language query in `input.json`, select the single most specific Jacobian operation that should handle it. Return the selected operation identifier. Write `submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier selects the Jacobian operation for the frozen query by a deterministic registry rule and checks the submitted operation identifier.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
