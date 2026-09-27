# Weyl exponents

`root_system.weyl_exponents.compute` returns the exact exponent sequence of
each connected component of a finite crystallographic Cartan datum. Each
component retains its simple-root indices, so reducible data preserve their
factorization and repeated exponent values across factors.

For an irreducible finite root system, the number of positive roots of height
`k` equals the number of Weyl exponents at least `k`. The operation takes
successive differences of these height counts to recover the exponents. This
classical duality is described, for example, in [Viswanath's note on exponents
and root heights](https://arxiv.org/abs/math/0609248). The operation does not
enumerate the Weyl group. The accepted input has rank at most 8 and the
existing complete positive-root bound of 120.

For example, type `G2` returns `(1, 5)`, while `A1 × A2` returns separate
components `(1)` and `(1, 2)` on their respective simple-root axes.
