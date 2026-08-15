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
three-dimensional Navier–Stokes equations. Claim only `COMPUTED` for the frozen
symbolic contract.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
