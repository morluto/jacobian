# Audit two degree-sequence claims

For each sorted sequence in `/app/input.json`, determine whether it is the
degree sequence of a finite simple undirected graph. For the graphical case,
submit any simple edge list whose exact degrees match the sequence. Edge pairs
may be oriented either way, and vertex labels may be zero-based (`0..n-1`) or
one-based (`1..n`). For the
nongraphical case, submit every violating Erdős–Gallai index `k`, with the
exact left and right sides of the inequality.

Bind a concise explanation at `/app/evidence/answer.txt` and write
`/app/submission.json`. Do not claim completeness beyond these two frozen
sequences or claim `VERIFIED`; the checker provides exact finite computation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
