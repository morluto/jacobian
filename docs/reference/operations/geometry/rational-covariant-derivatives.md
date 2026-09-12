# Rational coordinate covariant derivatives

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`differential_geometry.rational_tensor.covariant_derivative.compute` applies
the Levi--Civita connection of one exact rational coordinate metric to one
exact rational tensor over the same ordered chart. It returns a
`RationalCoordinateTensor` with one leading covariant derivative index; the
remaining variance and component axes are preserved in lexicographic order.

For

```text
T^(i_1 ... i_r)_(j_1 ... j_s)
```

the returned component is the exact rational-function identity

```text
(nabla_a T)^(i_1 ... i_r)_(j_1 ... j_s)
 = partial_a T^(i_1 ... i_r)_(j_1 ... j_s)
 + sum_u,b Gamma^(i_u)_(a b)
     T^(i_1 ... b ... i_r)_(j_1 ... j_s)
 - sum_v,b Gamma^b_(a j_v)
     T^(i_1 ... i_r)_(j_1 ... b ... j_s).
```

The metric and tensor must carry exactly the same coordinate axis and rational
field. A rank-zero tensor therefore receives only the leading covariant axis
and agrees with the rational-function gradient. Contravariant source indices
use the positive connection term; covariant source indices use the negative
term. A bare nested array is not interchangeable with this carrier because
variance and coordinate order change the mathematical result.

## Example: Euclidean polar coordinates

On the chart `(r, theta)`, use

```text
g_rr = 1,  g_rtheta = g_thetar = 0,  g_thetatheta = r^2.
```

The retained nondegenerate locus is `r != 0`, and the only nonzero connection
families are

```text
Gamma^r_thetatheta = -r,
Gamma^theta_rtheta = Gamma^theta_thetar = 1/r.
```

Applying the operation to `g` returns eight explicit zero components. For a
constant contravariant radial vector `V^r = 1`, `V^theta = 0`, the component
`(nabla_theta V)^theta` is `1/r`; for the covector `dr`,
`(nabla_theta dr)_theta` is `r`. These exact identities distinguish the
connection correction terms from componentwise differentiation.

## Exactness and execution envelope

The operation retains the metric and tensor denominator guards, generated
determinant guards, and every nonconstant denominator in the normalized output.
It admits the complete dense source and result tensor, sparse polynomial
support, coefficient height, intermediate DAG work, and canonical exact
output before symbolic DAG expansion. Rank-shape admission currently follows
canonical rational-function recognition, so a valid source whose derivative
would exceed rank can still perform bounded GCD recognition work first. A
request that exceeds the rank, component, term, coefficient, or
retained-locus envelope is rejected before symbolic/DAG expansion; accepted
requests return every component. The semantic limits are mathematical
work and exact representation limits, not a transport-byte truncation.

The exact symbolic arithmetic runs behind a bounded worker. Timeout,
cancellation, malformed backend output, or normalization failure is an
operational error and never establishes a zero or partial derivative. No
coordinate sampling, expression-string parser, generic manifold model, index
raising/lowering, tensor contraction, geodesic integration, or PDE evolution is
part of this operation.

For native use, import `covariant_derivative` from
`jacobian.math.geometry.differential.rational_tensor`; for MCP, inspect the
operation with `math.find` and pass the canonical tensor payload unchanged.
