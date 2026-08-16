# Certify an algebraic divisor class on a six-point blow-up

Submit a nonzero homogeneous cubic over `ZZ`, in the frozen ten-monomial
order, that vanishes simply at all six frozen points. For each point report the
polynomial value and three first partial derivatives. At least one derivative
must be nonzero, certifying multiplicity exactly one.

For the strict-transform divisor class `D=3H-E1-...-E6`, report its class
vector, `D^2`, `D·K` for `K=-3H+E1+...+E6`, and the adjunction arithmetic
genus. The verifier recomputes all evaluations and intersection arithmetic.

The coefficients must be primitive (i.e., their GCD must be 1); scalar multiples
Lefschetz (1,1) is a declared trusted theorem.
This one divisor certificate does not address higher-codimension Hodge classes
or prove the Hodge Conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
