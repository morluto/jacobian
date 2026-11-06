# Classify the nonnegative image of a cubic form

Determine exactly which nonnegative integers occur as
`A^3+B^3+C^3-3ABC` for nonnegative integers `A,B,C`.

Submit a complete certificate containing: the linear-times-quadratic
factorization; the full modulo-9 image obtained from all residue triples; the
excluded residues; and affine one-parameter constructions covering every
remaining residue class, including zero and their valid parameter domains.
Represent each affine formula `u*k+v` as `[u,v]`.

Use the factorization fields `linear: "A+B+C"` and
`quadratic: "A^2+B^2+C^2-AB-AC-BC"`.

A list of residue classes or a few numerical witnesses is insufficient. The
verifier independently expands every family symbolically, checks
nonnegativity, and proves residue-class coverage.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result. The finite certificate envelope permits at most 18 affine families (twice the public modulus), parameter_min in [0,500], and affine integer components of absolute value at most 2^256-1. Residue lists contain at most nine distinct elements of [0,8]. This bounds symbolic cubic expansion to degree three and the existing finite coverage replay to at most 18*501 parameter instances. Integer fields use exact JSON-number values: 2, 2.0, and 2e0 are equivalent. Numeric tokens are limited to 4300 characters, expanded integers to 4300 digits, and decimal exponents to [-4300,4300] before conversion, in addition to the tighter magnitude bounds above. The complete UTF-8 submission file, including whitespace, is limited to 16777216 bytes. All residue arrays represent sets; order does not affect correctness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
