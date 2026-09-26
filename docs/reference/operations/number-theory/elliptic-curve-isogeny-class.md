# Finite-field elliptic-curve isogeny classes

`elliptic_curve.finite_field.isogeny_class.decide` compares two nonsingular
short-Weierstrass curves over the same exact finite-field presentation. It
computes each point count by an exact quadratic-character sum, derives the
Frobenius trace `t = q + 1 - #E(F_q)`, and returns the characteristic polynomial
`X^2 - tX + q`. The result decides that the curves are isogenous over that
finite field exactly when these polynomials agree.

This equivalence is Tate's isogeny theorem for elliptic curves over finite
fields: [Tate, *Endomorphisms of Abelian Varieties over Finite Fields*](https://doi.org/10.1007/BF01404549).
The operation compares field presentations exactly; isomorphic presentations
require an explicit transport before comparison. It does not construct an
isogeny or claim that equal point counts imply curve isomorphism.

Both traces are admitted before either character sum. The field order is at
most 4096, and the paired exact-work bound is 8,000,000 units. The operation
uses the same exhaustive exact trace method as finite-field extension counts;
there is no backend guess or Hasse-compatible caller-supplied trace.

[Finite-field elliptic extension counts](elliptic-curve-extension-counts.md) ·
[Number theory operations](index.md) · [Operation reference](../index.md)
