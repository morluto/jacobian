# Coxeter polynomials

`root_system.coxeter_polynomial.compute` returns the characteristic polynomial
`det(tI-c)` of the Coxeter element on the simple-root lattice. For the ordered
simple-root axis `0,...,r-1`, the operation applies `s_0`, then `s_1`, through
`s_(r-1)`; as a matrix product this is `S_(r-1)...S_1 S_0`. This convention
matches Jacobian's left-to-right Weyl-word action. The result is a canonical
dense integer polynomial in descending degree order; the variable is `t`.

The Cartan datum is validated and the rank-bounded matrix product, subset
determinant work, coefficient magnitude, and output size are admitted before
expansion. The operation supports finite crystallographic Cartan data through
rank 8, including reducible products. It computes the characteristic polynomial
of this specified Coxeter element; it does not assert independence from a
different reflection order for arbitrary reducible data.

Fixtures include `A1: t+1`, `A2: t^2+t+1`, `B2: t^2+1`, and
`A1 x A2: (t+1)(t^2+t+1)`. Tests independently evaluate the returned
polynomial and compare it with a direct permutation determinant of `tI-c`.

[Operation references](../index.md) · [Tool surface](../../tools.md)
