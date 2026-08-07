# Construct an exact steady incompressible polynomial flow

Construct a non-irrotational steady two-dimensional polynomial velocity field
`u=(u1,u2)` with affine components and a quadratic pressure `p`, all over
`QQ`, such that `div(u)=0` and `(u·grad)u + grad(p) - Δu = 0`.

Use the monomial orders frozen in `/app/input.json`. Submit every coefficient
as a canonical rational string (`0`, `-3`, `5/7`; reduced denominator positive)
and include the independently checkable coefficient vectors for divergence,
both momentum residuals, and scalar vorticity `∂x u2 - ∂y u1`. The vorticity
must be nonzero, so the zero field and pure-gradient shortcuts are rejected.

`evidence/answer.txt` must be a JSON object with exactly `schema_version`,
`task_id`, `result`, and `limitations`. Use schema version `1` (the string
`"1"`), the task ID and limitations from the submission contract, and the same
`result` object as in `submission.json`. 
This finite symbolic certificate concerns one exact steady polynomial flow.
It neither proves nor disproves global existence or smoothness for the
three-dimensional Navier–Stokes equations. Claim only `CHECKED` for the frozen
symbolic contract.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact symbolic replay of one 2D polynomial field; no conclusion about the open 3D regularity problem.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `STEADY_INCOMPRESSIBLE_POLYNOMIAL_CERTIFICATE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
