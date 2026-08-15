# Audit an inverse-distance remainder claim

The frozen input contains a proposed disproof of a cubic remainder claim for
the inverse-distance expansion. Audit both the mathematical claim and the
reasoning in the frozen response.

Submit a complete exact certificate that:

1. gives the correct second-order term in invariant form;
2. uses rotational invariance to normalize `x=e_1`, then supplies two rational
   unit directions for `y=t*u`, one with a positive and one with a negative
   nonzero quadratic residual coefficient (coordinates are in this normalized
frame);
3. records each exact quadratic coefficient, thereby showing that the residual
   after the linear approximation is not generally `O(t^3)`; and
4. distinguishes the response's correct final conclusion from any invalid
   intermediate order estimate.

Directions may have dimension two, three, or four. They need not match the
Oracle directions. Rational numbers use reduced numerator/positive-denominator
objects. Write the structured result to `/app/submission.json`; it must display
the normalized quadratic term, both signed directional coefficients, and the
two order-analysis defects it identifies.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
