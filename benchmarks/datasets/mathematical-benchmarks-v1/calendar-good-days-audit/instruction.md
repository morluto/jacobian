# Audit a finite calendar claim

Audit the claim in the offline input by exhaustively checking the declared
finite date range. Return the truth value, exact count, and every qualifying
date in calendar order, including each concatenated integer.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier recomputes every qualifying date and concatenated value from the frozen input.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
