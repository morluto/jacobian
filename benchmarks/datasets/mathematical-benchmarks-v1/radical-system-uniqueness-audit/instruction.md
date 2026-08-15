# Audit a radical system and certify its unique real solution

The frozen input contains a real system involving square, cube, and fourth
roots, together with a claim that it has at least two solutions. Determine the
actual real solution set and audit that claim.

Submit a certificate that derives and independently checks an exact
univariate elimination polynomial, classifies every real root against the
principal-root domain constraints, reconstructs every surviving `(a,b,c)`
triple, and checks all three original equations exactly. You may choose the
valid elimination parameterization and algebraic route.
Your evidence must explain why rejected algebraic roots cannot represent real
solutions. Write `submission.json` according to `submission_schema.json` and
bind `evidence/answer.txt` by SHA-256.

This task has no external proof-assistant replay, so claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
