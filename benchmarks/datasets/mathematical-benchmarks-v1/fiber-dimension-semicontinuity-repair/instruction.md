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

The digest-bound `evidence/answer.txt` must contain exactly five lines:

1. `fiber-dimension-fitting-repair-v1`
2. `tensor-repair: <submitted tensor_repair>`
3. `global-repair: <submitted global_repair>`
4. `generator-count: <number of submitted ideal_generators>`
5. `fiber-dimensions: <points sorted lexicographically as x,y:dimension and joined by semicolons>`

Use conclusion `SEMICONTINUITY_PROOF_REPAIRED`, scope
`FROZEN_AFFINE_PRESENTATION_AND_IDENTIFIED_PROOF_GAPS`, completeness `COMPLETE`,
and claimed assurance `COMPUTED`. Include exactly this limitation:
`The verifier checks the frozen affine presentation and the two identified proof obligations; it does not formalize the full scheme-theoretic semicontinuity theorem.`

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier reports proof repair, symbolic ideal equality, fiber ranks, evidence, input binding, scope, completeness, limitations, protocol, and assurance separately. Sparse polynomials use canonical rational coefficients and bounded two-variable exponent vectors; duplicate monomials and zero terms are rejected.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SEMICONTINUITY_PROOF_REPAIRED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
