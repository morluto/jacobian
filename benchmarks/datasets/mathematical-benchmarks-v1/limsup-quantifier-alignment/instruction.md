# Audit a limsup formalization

The intended statement has the shape `∃ A, limsup X(A) ≤ Y`. A proposed formalization has the shape `∀ A, limsup X(A) ≥ Y`.

Determine their semantic relationship. Supply two finite exact-rational model families of possible limsup values:

1. one where the intended statement is true and the proposed statement is false;
2. one where the proposed statement is true and the intended statement is false.

For each family, report the truth values of both formulas and identify a witness for the existential or a violating witness for the universal. Values must be structured `{numerator, denominator}` objects within the frozen bounds. Equivalent encodings such as `2/8` and `1/4` are accepted after exact `Fraction` normalization. The verifier recomputes every comparison and accepts any valid separating families.

Write `/app/submission.json` using the supplied schema. The structured result
must describe the existential, universal, and incomparable relationship through
the two separating models.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
