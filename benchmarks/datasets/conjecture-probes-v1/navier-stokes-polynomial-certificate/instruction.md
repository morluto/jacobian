# Construct an exact steady incompressible polynomial flow

Construct a non-irrotational steady two-dimensional polynomial velocity field
`u=(u1,u2)` with affine components and a quadratic pressure `p`, all over
`QQ`, such that `div(u)=0` and `(u·grad)u + grad(p) - Δu = 0`.

Use the monomial orders frozen in `/app/input.json`. Submit every coefficient
as a structured `{numerator, denominator}` object. Equivalent encodings such as
`10/14` and `5/7` are accepted after exact `Fraction` normalization. Include
the independently checkable coefficient vectors for divergence,
both momentum residuals, and scalar vorticity `∂x u2 - ∂y u1`. The vorticity
must be nonzero, so the zero field and pure-gradient shortcuts are rejected.

This finite symbolic certificate concerns one exact steady polynomial flow.
It neither proves nor disproves global existence or smoothness for the
three-dimensional Navier–Stokes equations.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
