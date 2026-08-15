# Audit a Newton-polygon factorization lemma

The frozen input states the hypotheses and conclusion of an erroneous
factorization lemma from the first version of a paper, together with the kind
of left-edge control added in the correction.

Construct two nonconstant integer polynomials whose product satisfies every
old hypothesis at submitted indices `ell < j`, while both factor constant
terms have strictly positive `p`-adic valuation, contradicting the old
conclusion. Each factor must have degree at least two, their product must have
degree at least six, and `ell` must be at least two.

The product's lower Newton polygon must contain the primitive negative-slope
edge from `(ell,v(a_ell))` to `(j,0)` required by the old lemma and at least one
different negative-slope edge to its left. Your witness must also demonstrate
that at least one added left-edge condition in the corrected statement fails,
so it does not refute the repair.

Coefficient arrays are in ascending degree order and use canonical decimal
integer strings.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
