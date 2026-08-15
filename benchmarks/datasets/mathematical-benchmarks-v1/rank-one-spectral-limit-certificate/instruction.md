# Certify the limiting determinant root
Read `/app/input.json`. For the displayed `(n-1)×(n-1)` matrix, derive a diagonal-minus-rank-one decomposition, an exact partial-fraction identity for `1/(k^3-k)`, the resulting closed form for the unique determinant root, and its limit.

The requested outcome is the exact root formula for the declared matrix family and its limit; expanded determinant coefficients are outside scope.
Encode the root formula by its integer coefficient and the two affine offsets
in each of its numerator and denominator products.

Choose at least three distinct checkpoint values of `n` in the declared bounds and report both the reduced reciprocal sum and determinant root at each checkpoint. Supply the partial-fraction coefficients rather than a prose label.


<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/spectral-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
