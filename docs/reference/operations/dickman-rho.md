# Certified Dickman rho enclosures

[Documentation home](../../index.md) · [Operation references](index.md) · [Tool surface](../tools.md)

`number_theory.dickman_rho.piecewise_enclosure.compute` returns a reusable
pointwise enclosure of the Dickman--de Bruijn function through a requested
nonnegative rational endpoint `U`. The returned partition covers the complete
integer interval `[0, ceil(U)]`, so a caller can evaluate or integrate at any
point up to and including the final integer without rerunning the delay
equation.

Each piece `[n,n+1]` carries the explicit affine axis

```text
u = (n + 1/2) + t/2,       -1 <= t <= 1.
```

The coefficient tuple is in increasing powers of `t`. Every coefficient is a
dyadic ball and `uniform_remainder` bounds the central-polynomial truncation
error uniformly on the whole interval. Consequently, an evaluation at `t`
uses the coefficient-ball interval plus the signed remainder interval; no
backend or recurrence call is needed downstream.

The exact kernel starts with `rho(u)=1` on `[0,1]`. On `[n,n+1]`, the delay
equation becomes

```text
(2*n + 1 + t) p'(t) + p_previous(t) = 0.
```

The denominator is at least `2*n` for every `t` in `[-1,1]` and `n >= 1`.
After matching coefficients through degree `d-1`, the only residual term is
`(d*p[d] + p_previous[d]) t^d`; integrating it with the denominator lower
bound gives the published uniform remainder. Dyadic outward rounding is added
to the pointwise width separately.

`precision_bits` is bounded and controls coefficient and remainder rounding.
The operation admits the interval count, all degree candidates through the
published degree ceiling, exact coefficient-growth estimate, and result carrier
before expanding the recurrence. Requests whose target width cannot fit the
bounded degree or precision are typed resource-admission failures.

The operation is an enclosure, not the distinct de Bruijn asymptotic
approximation. The exact controls are `rho=1` on `[0,1]`,
`rho(u)=1-log(u)` on `[1,2]`, recurrence residual identities, positivity and
monotonicity where the requested width certifies them, independent high-
precision overlap tests, and narrowing as precision increases.
