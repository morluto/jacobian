# Audit a prime-to-prime-power inference

The frozen proof partitions a sum into `m/p` cycle sums for every prime `p | m`. It establishes
that each cycle sum is divisible by `p`, then concludes that the total is divisible by `m`.

Submit a compressed countermodel to that inference. Choose a prime `p` and exponent `e >= 2`, set
the compatibility field `r = 1`, and use `m = p^e <= 10000` with at least four cycles. Give two to
six distinct cycle-sum groups. Each group records a positive multiplicity and a positive cycle sum;
the multiplicities must total `m/p`, every cycle sum must be divisible by `p`, and at least two
different cycle sums must occur. The resulting total must be divisible by `p` but not by `m`.

Report the exact p-adic valuations of `m` and of the total, the local/global statuses, and the
missing proof obligation. Write `/app/submission.json` and one digest-bound JSON task-specific witness file at
`evidence/divisibility-audit.json`. The task-specific witness object must contain exactly
`schema_version: "1"`, the task ID, and the result, with the result matching
the submission. This audit invalidates one proof step and does not
disprove the source theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/divisibility-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
