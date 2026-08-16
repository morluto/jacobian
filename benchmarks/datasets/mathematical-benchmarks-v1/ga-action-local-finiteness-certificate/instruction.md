# Construct a finite invariant-subspace certificate

The frozen input defines the additive-group action

`alpha_t(x,y) = (x + t*y, y)`

on `QQ[x,y]` and a homogeneous degree-four polynomial `f`. Construct an
ordered five-element rational polynomial basis for a finite-dimensional
subspace containing `f`, together with coordinates of `f` and the exact
polynomial action matrix in that basis.

Columns encode images: if `B=(b_0,...,b_4)`, entry `(i,j)` is the coefficient
of `b_i` in `alpha_t(b_j)`. All sparse term lists must use nonzero structured
`{numerator, denominator}` rational coefficients, unique exponent tuples, and
ascending exponent order. Equivalent encodings such as `2/2` and `1` are
accepted after exact `Fraction` normalization. The basis polynomials must be
homogeneous of total degree four and linearly independent.

Your certificate must make it possible to check all of the following without
trusting a preferred basis:

1. the submitted coordinates reconstruct the frozen `f`;
2. every substituted basis polynomial equals the corresponding matrix column;
3. `R(0)=I`;
4. `R(s+t)=R(s)R(t)` as an exact polynomial identity.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
