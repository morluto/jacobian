# Repair a fiber-dimension semicontinuity proof

The frozen input summarizes two known invalid steps from a generated proof of
upper semicontinuity. Apply the supplied repair obligations to the affine
presentation and construct the exact determinantal certificate. This task
checks certificate construction from disclosed diagnoses; it does not measure
blind diagnosis of the unavailable source proof.

For a presentation `A^3 -> A^2 -> F -> 0`, tensoring with a residue field only
needs right exactness to identify the fiber as the cokernel. A residue field is
not generally flat over its local ring. Global closedness must be expressed by
the determinantal/Fitting ideal, not by an arbitrary union of locally closed
sets.

For the frozen `2 x 3` matrix, compute a generating set for the ideal of all
`2 x 2` minors, which cuts out the locus where the cokernel fiber dimension is
at least one. You may submit any nonzero sparse rational polynomials generating
the same ideal; order and choice of generators are not prescribed. A sparse
term has a canonical rational `coefficient` and `exponents: [x_power,y_power]`.
Duplicate monomials within one polynomial are forbidden. Canonical rationals
are reduced integers or fractions with no leading zeros, exponent notation, or
denominator `1`.

For every point listed in `input.json`, submit the exact specialized matrix
rank and cokernel dimension. Each point must occur exactly once.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
