# Gaussian-moment generality audit

The frozen packet gives a three-real-Gaussian template and a formal Lagrange-inversion identity. A prior audit checked twelve moments and then claimed the identities for every exponent. That extrapolation is invalid.

Choose any nonzero rational parameter `a` within the declared bounds. Derive the quadratic `v(z)` and rational inverse branch `zeta(t)` satisfying the identities below; both are determined by `a` and the correction-factor identity rather than freely chosen. Submit canonical rational coefficient lists in ascending degree order as integer `numerator`/positive integer `denominator` objects in lowest terms.

Your certificate must establish, by exact rational-function identities rather than samples, all of the following:

1. `zeta = t h(zeta)` for `h(z)=a+z`;
2. `1 - 2 t v(zeta) = (1-t)^(-2)`;
3. the constant-term-one square-root branch cancels `1-t h'(zeta)`;
4. `E(exp(tP))=1` and `E(Q exp(tP))=t/(1-t)`;
5. therefore `E(P^m)=0` and `E(QP^m)=m!` for every `m>=1`.

Also classify the twelve-moment argument as insufficient: the submitted
rational-function identities, not finite checking, establish the all-exponents
statement.

Write `/app/submission.json` matching the supplied schema.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
