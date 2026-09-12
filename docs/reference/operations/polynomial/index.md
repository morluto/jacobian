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

## Rational coordinate maps

`rational_function_map.compose.compute` composes an outer map

\[
F : (y_1,\ldots,y_m) \longrightarrow (z_1,\ldots,z_r)
\]

with an inner map

\[
G : (x_1,\ldots,x_n) \longrightarrow (y_1,\ldots,y_m).
\]

The intermediate axis is a typed contract: the ordered
`inner.target_coordinates` must equal `outer.source_variables`. The exact
result retains both input maps, the composite map on the inner source axis,
and the outer target axis. It is a field identity over `QQ`, not a claim about
an inverse, image, or global chart.

The `construction_locus_guard` is an ordered, duplicate-free tuple of monic
`RationalPolynomial` values interpreted conjunctively as nonvanishing
conditions. It contains every nonconstant inner component denominator and the
numerator of each normalized substituted outer denominator, before any
removable cancellation in the returned composite. Thus a cancellation may
extend the canonical rational function while the result still preserves the
stricter locus on which the supplied composition was constructed. An outer
denominator that becomes the zero rational function is rejected.

Admission accounts for every source component, denominator-clearing
intermediate, normalization/GCD phase, guard, and complete canonical output
before exact backend expansion. The native entry point is
`jacobian.math.polynomials.rational_functions.composition.compose_maps(outer, inner)`;
native callers pass canonical `RationalFunctionMap` values rather than the
wire request model.

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

## Focused contracts

- [Monomial-ideal graded Betti profiles](monomial-ideal-graded-betti.md)
