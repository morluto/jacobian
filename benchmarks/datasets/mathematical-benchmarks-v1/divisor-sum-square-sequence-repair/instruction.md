# Construct infinitely many square divisor sums

The frozen solution claims no sequence exists, relying on probabilistic
language that does not apply to a deterministic existence problem. Repair it.

Submit a deterministic piecewise formula for positive integers `a_n` with
`a_1=1`, a default power-of-two branch for every index that is not an odd prime
(including `n=2`), and a separate odd-prime branch. Certify that for each fixed
positive `k`, every `a_n` with `n>=max(2,k)` is divisible by `2^k`, and
therefore only finitely many can equal `k mod 2^k`. Also submit at least four
freely chosen distinct odd-prime probes where `b_p = sum_{d|p} d*a_d` is an
exact square.
Encode the default power-of-two branch by its exponent offset. The verifier
checks each odd-prime branch semantically from the submitted exact probes; no
decorative formula or threshold-rule string is required.

Use `/app/submission.json` and bind an identical certificate at
(the task identifier), `result` (an object equal to

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
