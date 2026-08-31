# Polynomial operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The live catalog is the authoritative reference for installed polynomial
operations and their request/result schemas. The native API under
`jacobian.math.polynomials` exposes the same direct domain kernels for Python
callers.

Exact polynomial values retain their variable order, sparse terms, and
canonical rational coefficients. Operation-specific bounds are checked by the
shared domain admission path before SymPy is called; a wire request model may
invoke that path after parsing, while native callers use it directly. A bounded
result is returned inline; no polynomial is implicitly published or retained
for replay.

## Focused contracts

- [Monomial-ideal graded Betti profiles](monomial-ideal-graded-betti.md)
