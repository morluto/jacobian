# Construct an optimal distinct-sum pairing

For the frozen ground set, construct as many disjoint unordered pairs as
possible so that all pair sums are distinct and at most `n`.

Submit only the pairs, with the smaller member first in each pair and the
list of pairs in lexicographically increasing order. The verifier independently
checks the submitted pairing and
exhaustively solves the finite optimization problem; it accepts any optimal
pairing, not one expected arrangement.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The public ground set is the 15 integers from 1 through 15. Submit at most 7 pairs, with each member in that ground set. These packing and member bounds precede mathematical checks of disjointness, distinct sums, the sum limit, optimality, and canonical ordering.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
