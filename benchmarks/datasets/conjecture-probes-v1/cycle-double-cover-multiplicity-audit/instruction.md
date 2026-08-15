# Cycle-double-cover multiplicity audit

The frozen validator accepts any collection of simple cycles whose union covers every edge. This is not the cycle-double-cover contract, which requires every edge to occur exactly twice **counting cycle multiplicity**.

For the supplied Petersen graph, submit a complete certificate containing:

1. a flawed-pass collection of at least four distinct simple cycles that covers every edge but is not a double cover;
2. its complete 15-entry edge-multiplicity vector and the exact sorted indices whose multiplicity is not two; and
3. a repaired collection of distinct simple cycles whose complete multiplicity vector is exactly two on every edge.

Each cycle is a vertex list without a repeated closing vertex. The verifier canonicalizes rotations and reversals, checks simplicity and every consecutive graph edge, rejects duplicate cycles, and recomputes both multiplicity profiles. Alternative valid collections are accepted.

This is one finite Assurance audit. It does not establish the Cycle Double Cover Conjecture for any graph family. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
