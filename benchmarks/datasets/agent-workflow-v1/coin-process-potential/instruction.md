# Unit-decrement potential for the coin process

There are twelve coins in positions `1..12`; a state is a bit vector, with `1`
meaning heads.  If a nonzero state contains exactly `k` heads, one step flips
the coin in position `k`.  The all-tails state stops.

Find an integer potential of the form

`P(x) = sum_i w_i x_i + q * sum_{i<j} x_i x_j`

that decreases by exactly one on every legal step.  Certify its minimum on
each Hamming-weight layer, use it to prove termination for all `2^12` states,
and compute the exact total and average stopping time.

The verifier independently checks every state, transition, layer minimum, and
average from the structured certificate in `submission.json`.  Merely
reporting the published average is insufficient.  Submit
`/app/submission.json` following the schema and a nonempty human-readable
summary at `/app/evidence/answer.txt`.  The prose file is checked for path and
digest integrity only; mathematical scoring comes from the structured
certificate rather than keyword matching.

Do not claim proof-assistant verification.  This is an exact finite-state
certificate for `n=12` only.  The `limitations` array must contain exactly
this one entry: `This certificate applies only to the frozen 12-coin
instance (n=12).`

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `TERMINATION_AND_MEAN_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
