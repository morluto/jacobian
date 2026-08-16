# Repair an omitted Apollonius-circle computation

An OPC proof received human score zero after asserting, without computation, that the circle with diameter the internal/external ratio points implies `MA^2 = k^2 MB^2`.

Choose a positive structured rational `k != 1` and `c`. Put `A=(0,0)`, `B=(c,0)`, `P=kc/(k+1)`, and `Q=kc/(k-1)`. Submit `P,Q`, the center and positive radius of the diameter circle, and coefficient vectors in basis `[x^2,y^2,x,1]` for its expanded equation and for `MA^2-k^2 MB^2`. Submit the exact proportionality multiplier proving the omitted identity.

Every rational field is a `{numerator, denominator}` object. Equivalent encodings such as `2/2` and `1` are accepted after exact `Fraction` normalization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
