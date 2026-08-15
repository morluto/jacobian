# Repair a natural-subtraction proof

Audit the failed rewrite in the frozen natural-number proof branch, then give
an exact algebraic repair certificate.

First report whether the failed pattern occurs as a subtree of the target AST.
Then use the declared equation basis to derive the goal: submit one rational
multiplier per basis equation and the resulting coefficient vector in the
declared variable order. The subtraction-recovery equation is justified only
by the recorded `b<=a` side condition.

The verifier independently traverses the expression tree and recomputes the
linear combination over exact rationals. Write `submission.json` to
`submission_schema.json`, put a task-specific witness in `evidence/answer.txt`,
and bind its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
