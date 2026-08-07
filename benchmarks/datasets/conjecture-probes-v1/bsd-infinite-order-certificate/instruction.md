# Certify an infinite-order point without overclaiming BSD

Choose integers `A,B` and a nonzero integral point `P=(x,y)` within the bounds
in `/app/input.json` on the nonsingular curve `y^2=x^3+A*x+B`. Supply the exact
discriminant `-16(4A^3+27B^2)`, `y^2`, and the exact rational coordinates of
`2P` and `3P` using canonical reduced rational strings.

Your certificate must satisfy the Lutz–Nagell obstruction: `y != 0` and `y^2`
does not divide the absolute discriminant. The verifier independently checks
the curve equation, nonsingularity, divisibility obstruction, and both group
law computations. Lutz–Nagell itself is a declared trusted theorem.

`evidence/answer.txt` is a JSON object (`application/json`) with exactly
`schema_version` (the string `"1"`), `task_id`, `result`, and `limitations`,
matching the submission.

This proves only that one authored elliptic curve has a rational point of
infinite order. It does not compute an L-function, determine the full rank, or
prove any case of the Birch–Swinnerton-Dyer conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Checks one infinite-order certificate under trusted Lutz-Nagell; no Birch-Swinnerton-Dyer conclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `ELLIPTIC_POINT_INFINITE_ORDER_CERTIFICATE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
