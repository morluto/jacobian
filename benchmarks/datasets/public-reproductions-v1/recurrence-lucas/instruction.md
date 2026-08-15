# Operation discovery regression

Given the natural-language query in `input.json`, select the single most specific Jacobian operation that should handle it. Return the selected operation identifier. Write `submission.json` to the exact agent-visible `submission_schema.json`, record the selection in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence list. Claim `COMPUTED` assurance; do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit the result and the required replay artifact. The verifier recomputes the task-specific mathematical claim from the frozen input.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
