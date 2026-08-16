# Certify a sharp six-variable inequality

For positive `a,b,c,x,y,z` with `x+y+z=1`, determine the largest real
constant `C` such that

`a*x+b*y+c*z + C*sqrt((x*y+y*z+z*x)*(a*b+b*c+c*a)) <= a+b+c`.

Submit `C`, an exact symbolic upper-bound certificate in one of the
registered proof modes, and a positive rational equality witness proving
sharpness. In `CS_COMPOSITION` mode, provide the four quadratic forms used by
the two Cauchy--Schwarz steps and both exact sum-square residual identities. In
`AMGM_SQUARES` mode, first normalize the coefficient triple by its positive
sum, so `a+b+c=1`. In both `AMGM_SQUARES` and `DIRECT_SOS` modes, use the
following core polynomials exactly:

- `d = a*x+b*y+c*z`;
- `u = a*b+b*c+c*a`;
- `v = x*y+y*z+z*x`;
- `residual = 1-d-u-v`;
- `constraint_residual = 2-(a+b+c)^2-(x+y+z)^2`.

In `AMGM_SQUARES` mode, `sos_twice` is
`(a-x)^2+(b-y)^2+(c-z)^2`, and must satisfy
`2*residual-sos_twice=constraint_residual`. In `DIRECT_SOS` mode, provide a
list `sos_factors` of sparse polynomials whose squared sum equals
`2*residual-constraint_residual`; any independently checkable sum-of-squares
decomposition is accepted. Sparse polynomials use variables ordered
`[a,b,c,x,y,z]`, integer coefficients, and exponent vectors; term order is not scored. Numerical sampling and prose-only proofs are not accepted.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
