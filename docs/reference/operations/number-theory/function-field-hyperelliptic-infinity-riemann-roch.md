# Hyperelliptic Riemann-Roch space at infinity

`function_field.hyperelliptic_infinity_riemann_roch_space.compute` returns the
complete basis and dimension of `L(m P_infinity)` for the unique rational
infinity place of an admitted odd-degree squarefree model `y^2 = f(x)` over
`GF(p)`. The place and every returned element retain the same exact function
field. The signed multiplicity `m` is the coefficient of `P_infinity`; for
`m < 0` the space is zero.

If `d = deg(f) = 2g + 1`, then `v_infinity(x) = -2` and
`v_infinity(y) = -d`. Every function has a unique form `A(x) + B(x)y`. The
monomials in `A(x)` have even pole order and those in `B(x)y` have odd pole
order, so leading terms cannot cancel. Therefore a basis consists exactly of

```text
x^i       for 0 <= 2i <= m
x^j * y   for 0 <= 2j + d <= m
```

The operation checks the supported model and infinity-place parent first. It
admits the basis dimension, polynomial degree, estimated construction work,
and serialized result size before creating basis elements.

This operation covers multiples of the supported odd-degree infinity place.
It does not claim Riemann-Roch spaces for arbitrary divisors or even-degree
models, whose infinity places can split or have higher residue degree.

[Number-theory operations](index.md) · [Tool surface](../../tools.md)
