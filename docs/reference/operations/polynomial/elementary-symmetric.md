# Elementary-symmetric polynomial families

[Documentation home](../../../index.md) · [Polynomial operations](index.md) ·
[Tool surface](../../tools.md)

`polynomial.symmetric.elementary_family.compute` returns the complete family
`(e_0, ..., e_k)` of elementary symmetric polynomials for an ordered variable
axis `x_1, ..., x_n`.  Every returned member is the ordinary canonical sparse
`QQ` polynomial used by substitution, evaluation, polynomial maps, ideals, and
expression normalization.  `e_0` is the multiplicative constant `1`, including
when the axis is empty.

The axis is retained on every polynomial.  Renaming or permuting variables
therefore changes only the coordinates of the sparse exponent vectors: the
family is invariant under that coherent variable permutation.  The defining
recurrence is

```text
e_j(x_1, ..., x_n) = e_j(x_1, ..., x_(n-1))
                     + x_n e_(j-1)(x_1, ..., x_(n-1)).
```

## Bounded domain

The request has at most eight distinct variables and requires `0 <= k <= n`.
Before the recurrence runs, admission counts the exact

`sum(j=0..k) binomial(n,j)`

monomials, all exponent cells, repeated axis-label storage, recurrence work,
and canonical result cells.  The complete eight-variable family has 256
monomials and 2,048 exponent cells, well within the shared sparse polynomial
carrier.  These are semantic representation and work reservations; they are
not transport byte limits.

The kernel uses the dynamic-product recurrence once and materializes ordinary
canonical sparse polynomials.  It does not parse caller expressions or expose
backend-specific symbolic objects.  Complete homogeneous, power-sum, Schur,
monomial-symmetric, and basis-conversion operations remain separate contracts.
