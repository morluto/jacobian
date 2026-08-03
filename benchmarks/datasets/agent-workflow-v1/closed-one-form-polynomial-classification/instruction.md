# Classify an exact swapped polynomial one-form

Read `/app/input.json`. Let `f` have the ten coefficients in the declared order. Classify exactly those degree-at-most-three polynomials for which

`f(x,y) dx + f(y,x) dy`

has zero integral around every closed smooth curve in `R2`.

The frozen source solution claims that closedness is `f_y(x,y)=f_x(y,x)`. Audit the chain rule rather than trusting that claim.

Write `/app/submission.json` matching the supplied schema and `/app/evidence/derivation.txt`. Submit two independent integer row constraints on the ten coefficients, their rank and the resulting dimension, eight integer coefficient vectors forming a basis of the solution space, and one polynomial potential for each basis vector. A potential `F` must satisfy `F_x=f(x,y)` and `F_y=f(y,x)` exactly. Rational coefficients use canonical strings such as `1/2` or `-3`; combine like terms and omit zero terms.

The verifier independently derives the correctly chained closedness constraints, checks row-space equality rather than a fixed presentation, performs exact rank and nullspace tests, and differentiates every potential. It does not assess arbitrary smooth forms or prove the Poincare lemma; assurance must remain `COMPUTED`.
