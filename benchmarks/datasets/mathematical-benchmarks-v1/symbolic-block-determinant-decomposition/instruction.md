Construct a rational change of basis on the three block indices for the frozen symbolic matrix `C`.

Your first basis vector must span the common-coordinate channel and the other two must independently span the sum-zero channel. Submit the basis matrix and its inverse using canonical rational strings. The verifier will build arbitrary symbolic `2 x 2` matrices `A` and `B` and independently check the complete `6 x 6` similarity identity over `QQ[a11,...,b22]`.

Report the three resulting channels in their actual diagonal order, the determinant factorization, and whether the source proof's invertibility assumption is required for this polynomial identity. Do not claim that the general `n,k` theorem or the dataset proof was machine verified.

Write `/app/submission.json` and `/app/evidence/answer.txt`. The evidence file must contain one `RESULT_JSON:` line equal to the submitted result and briefly explain the common channel, sum-zero channels, exact similarity replay, and scope limitation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
