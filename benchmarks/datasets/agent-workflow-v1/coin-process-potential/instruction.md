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
average.  Merely reporting the published average is insufficient.  Submit
`/app/submission.json` following the schema and one bound explanation at
`/app/evidence/answer.txt`.

Do not claim proof-assistant verification.  This is an exact finite-state
certificate for `n=12` only.
