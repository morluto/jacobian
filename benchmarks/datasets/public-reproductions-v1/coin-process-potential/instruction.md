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
reporting the published average is insufficient.  Submit `/app/submission.json` following the schema.
Do not claim proof-assistant verification.  This is an exact finite-state
certificate for `n=12` only.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
