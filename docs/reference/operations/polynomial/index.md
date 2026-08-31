# Polynomial operations

## Rational polynomial ideals

`polynomial.ideal.containment.decide` decides the directed relation
`I subseteq J` in one exact ordered polynomial ring over `QQ`. Its computed
result retains the source and target presentations and a source-ordered
Gröbner normal-form ledger. A positive result covers every source generator;
a negative result ends at the first nonzero normal form, which is an exact
obstruction to containment.

`polynomial.ideal.equality.decide` computes both directed ledgers under one
request deadline and reports equality exactly when both containments hold.
The conclusion therefore does not depend on generator order, redundant
generators, or multiplication of generators by nonzero rational scalars.
Both operations support `lex`, `grlex`, and `grevlex`; the selected order is
retained because the normal-form witnesses depend on it even though the ideal
relation does not.

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
