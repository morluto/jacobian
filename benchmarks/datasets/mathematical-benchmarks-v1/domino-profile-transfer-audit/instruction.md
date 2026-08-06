# Repair a corner-deficient domino tiling count

Audit the proposed remainder for a `3 x 2021` board with one corner removed.
Use the frozen three-bit profile semantics, but derive the state transitions
yourself. Submit `/app/submission.json` following the public schema and
digest-bound prose at `/app/evidence/answer.txt`.

Your certificate must include the chosen missing corner row, the complete
`8 x 8` transition matrix modulo 19, the initial profile vector, and the full
sequence of vector updates for every set bit in the binary exponentiation of
2021. Each update records its bit index and vectors before and after applying
the corresponding matrix power. Give the repaired remainder and explain the
failure in the proposed decomposition.

The verifier regenerates legal domino placements rather than trusting the
matrix. Claim `COMPUTED` assurance and complete scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently reconstructs and replays the exact profile-transfer certificate.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PROPOSED_REMAINDER_INCORRECT_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
