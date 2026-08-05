# Audit a prime-to-prime-power inference

The frozen proof partitions a sum into `m/p` cycle sums for every prime `p | m`. It establishes
that each cycle sum is divisible by `p`, then concludes that the total is divisible by `m`.

Submit a compressed countermodel to that inference. Choose a prime `p`, exponent `e >= 2`, and a
positive factor `r` coprime to `p`, with `m = p^e r <= 10000` and at least four cycles. Give two to
six distinct cycle-sum groups. Each group records a positive multiplicity and a positive cycle sum;
the multiplicities must total `m/p`, every cycle sum must be divisible by `p`, and at least two
different cycle sums must occur. The resulting total must be divisible by `p` but not by `m`.

Report the exact p-adic valuations of `m` and of the total, the local/global statuses, and the
missing proof obligation. Write `/app/submission.json` and one digest-bound JSON evidence file at
`evidence/divisibility-audit.json`. The evidence must repeat exactly the task ID, result, and
limitations. Maximum assurance is `COMPUTED`; this audit invalidates one proof step and does not
disprove the source theorem.
