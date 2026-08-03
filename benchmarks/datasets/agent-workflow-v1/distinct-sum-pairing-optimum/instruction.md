# Construct an optimal distinct-sum pairing

For the frozen ground set, construct as many disjoint unordered pairs as
possible so that all pair sums are distinct and at most `n`.

Submit the pairs in canonical increasing order, their sums, and the claimed
optimum. The verifier independently checks the witness and exhaustively solves
the finite optimization problem; it accepts any optimal pairing, not one
expected arrangement. Write `submission.json` to `submission_schema.json`,
explain the five-pair construction, its distinct sums, and the exhaustive
exclusion of a six-pair solution in `evidence/answer.txt`, and bind its SHA-256
digest.
