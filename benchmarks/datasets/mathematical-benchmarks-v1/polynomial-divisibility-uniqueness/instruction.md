Determine every integer parameter `a` for which

`x^2 - x + a` divides `x^13 + x + 90` in `Z[x]`.

Write `/app/submission.json` using `/app/submission_schema.json` and a concise derivation at `/app/evidence/answer.txt`. Coefficient arrays are ascending. Your certificate must give the two coefficient polynomials of the remainder in `Q[a][x]`, their monic gcd in `Q[a]`, the unique parameter, and the exact integer quotient at that parameter. The gcd is what establishes completeness over all integers; checking only the reported parameter is insufficient.

Include one line beginning `RESULT_JSON:` in the evidence file, followed by the JSON serialization of the submitted `result` object. Also state the concrete remainder coefficient arrays, gcd, parameter, and quotient in the derivation.

Report `COMPUTED`, not `VERIFIED`. The verifier independently reconstructs the symbolic remainder, polynomial gcd, and product identity.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `UNIQUE_PARAMETER`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
