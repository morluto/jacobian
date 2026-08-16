# Certify a finite Littlewood-product minimum

For every integer `1 <= n <= 2000`, consider
`n ||n sqrt(2)|| ||n sqrt(3)||`. Use rigorous rational lower and upper
enclosures obtained from the frozen 80-digit integer-square-root scale, never
floating point. Submit the complete strict record-minimum sequence. Each row
must include `n`, the two floor values, the two nearest integers, and structured
`{numerator, denominator}` rational lower/upper bounds for the product. Identify the final finite argmin.

The verifier independently reconstructs all 2000 enclosures and the complete
record sequence. 
This certifies only the frozen finite range for `(sqrt(2),sqrt(3))`; it does not
establish a liminf or any case of Littlewood's conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
