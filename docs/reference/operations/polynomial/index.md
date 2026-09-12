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
owning domain admission path before the backend is called. MCP parses the
wire request before invoking the domain function; native callers use the same
function directly. A bounded result is returned inline; no polynomial is implicitly published or retained
for replay.

## Typed expression normalization

`polynomial.expression.normalize` expands one bounded, exact AST into a
canonical `RationalPolynomial`. Its closed grammar is deliberately small:

- `LITERAL` carries one reduced `CanonicalRational`;
- `VARIABLE` names one member of the ordered `variables` axis;
- `ADD` and `MULTIPLY` carry finite operand tuples; and
- `POWER` carries a nonnegative bounded integer exponent.

The AST is a value, not a source-language string. Division, negative powers,
function calls, assumptions, and textual parser syntax are outside the contract.
The operation admits node/depth, support, degree, exact coefficient-height,
intermediate work, and coefficient-representation bounds before expansion. The
result retains the source value and its explicit coefficient domain/variable
axis, so callers can compare normalized values or pass the polynomial directly
to another polynomial operation.

## Focused contracts

- [Monomial-ideal graded Betti profiles](monomial-ideal-graded-betti.md)
