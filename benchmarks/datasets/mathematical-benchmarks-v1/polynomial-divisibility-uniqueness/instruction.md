Determine every integer parameter `a` for which

`x^2 - x + a` divides `x^13 + x + 90` in `Z[x]`.

Write `/app/submission.json` using `/app/submission_schema.json`. Coefficient
arrays are ascending. Your certificate must give the two coefficient
polynomials of the remainder in `Q[a][x]`, their monic gcd in `Q[a]`, the
unique parameter, and the exact integer quotient at that parameter. The gcd is
what establishes completeness over all integers; checking only the reported
parameter is insufficient. The verifier independently reconstructs the
symbolic remainder, polynomial gcd, and product identity.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result. The finite certificate envelope permits at most 14 coefficients in each polynomial, including trailing zero padding. The integer parameter lies in [-90,90]: any divisor constant must divide the nonzero dividend constant 90. Every coefficient has absolute value at most sum(abs(dividend))*(abs(dividend[0])+2)^(degree(dividend)+1) = 286297404070204943107692167168. This public-input bound dominates the residue recurrence and monic quadratic-division recurrence without exposing their computed coefficients. Integer fields use exact JSON-number values: 2, 2.0, and 2e0 are equivalent. Numeric tokens are limited to 4300 characters, expanded integers to 4300 digits, and decimal exponents to [-4300,4300] before conversion, in addition to the tighter magnitude bounds above. The complete UTF-8 submission file, including whitespace, is limited to 1048576 bytes.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
