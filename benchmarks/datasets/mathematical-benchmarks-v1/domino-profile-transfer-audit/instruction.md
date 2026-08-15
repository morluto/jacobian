# Repair a corner-deficient domino tiling count

Audit the proposed remainder for a `3 x 2021` board with one corner removed.
Use the frozen three-bit profile semantics, but derive the state transitions
yourself. Submit `/app/submission.json` following the public schema and a
digest-bound JSON envelope at
`/app/evidence/profile-transfer-certificate.json`. The envelope must contain
with the latter three matching the submission.

Your certificate must include the chosen missing corner row, the complete
`8 x 8` transition matrix modulo 19, the initial profile vector, and the full
sequence of vector updates for every set bit in the binary exponentiation of
2021. Each update records its bit index and vectors before and after applying
the corresponding matrix power. Give the repaired remainder and

The verifier regenerates legal domino placements rather than trusting the

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/profile-transfer-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
