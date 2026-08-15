Determine every integer parameter `a` for which

`x^2 - x + a` divides `x^13 + x + 90` in `Z[x]`.

Write `/app/submission.json` using `/app/submission_schema.json` and a concise derivation at `/app/evidence/answer.txt`. Coefficient arrays are ascending. Your certificate must give the two coefficient polynomials of the remainder in `Q[a][x]`, their monic gcd in `Q[a]`, the unique parameter, and the exact integer quotient at that parameter. The gcd is what establishes completeness over all integers; checking only the reported parameter is insufficient.

Include one line beginning `RESULT_JSON:` in the evidence file, followed by the JSON serialization of the submitted `result` object. Also state the concrete remainder coefficient arrays, gcd, parameter, and quotient in the derivation.

Report `COMPUTED`, not `VERIFIED`. The verifier independently reconstructs the symbolic remainder, polynomial gcd, and product identity.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
