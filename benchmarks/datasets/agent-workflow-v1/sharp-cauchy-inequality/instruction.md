# Certify a sharp six-variable inequality

For positive `a,b,c,x,y,z` with `x+y+z=1`, determine the largest real
constant `C` such that

`a*x+b*y+c*z + C*sqrt((x*y+y*z+z*x)*(a*b+b*c+c*a)) <= a+b+c`.

Submit `C`, an exact symbolic upper-bound certificate in one of the
registered proof modes, and a positive rational equality witness proving
sharpness. In `CS_COMPOSITION` mode, provide the four quadratic forms used by
the two Cauchy--Schwarz steps and both exact sum-square residual identities. In
`AMGM_SQUARES` mode, first normalize the coefficient triple by its positive
sum, then provide the normalized polynomial residual and its exact
sum-of-squares decomposition. In `DIRECT_SOS` mode, provide the core
polynomials `d`, `u`, `v`, the `residual`, the `constraint_residual`, and a
list `sos_factors` of sparse polynomials whose squared sum equals
`2*residual - constraint_residual`; any independently checkable sum-of-squares
decomposition is accepted. Sparse polynomials use variables ordered
`[a,b,c,x,y,z]`, integer coefficients, and lexicographically ordered exponent
vectors. Numerical sampling and prose-only proofs are not accepted.

Write `submission.json` and digest-bind `evidence/inequality-certificate.json`,
which must copy `result` and `limitations` exactly. Claim at most `COMPUTED`.
