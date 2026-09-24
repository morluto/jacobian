# Polynomial operations

## Holonomic formal series

`holonomic.differential_series.construct` represents the unique formal Taylor
series at \(x=0\) determined by a nonzero differential operator over
\(\mathbb{Q}(x)\) and the exact initial derivatives
\(f(0),\ldots,f^{(r-1)}(0)\), where (r) is the operator order. Every
coefficient must be regular at the center and the leading coefficient must be
nonzero there. This ordinary-point condition lets coefficient comparison
determine each later Taylor coefficient uniquely. The returned
`DFinitePowerSeries` retains the operator, initial derivatives, and center as
one composable value. It denotes a formal series and makes no convergence or
analytic-continuation claim. The ordinary-point coefficient-comparison
principle is described in [DLMF §2.7](https://dlmf.nist.gov/2.7). Singular-center
solutions require separate semantics.

`holonomic.differential_series.generate_finite_prefix.compute` returns the
first `count` Taylor coefficients as a `FiniteRationalSequence`. Its entries
are coefficients of powers of \(x\), so the supplied initial derivatives are
divided by \(i!\) at indices below the operator order. It accepts polynomial
differential coefficients in \(\mathbb{Q}[x]\); rational-function coefficient series
remain representable but are outside this prefix operation's current domain.
The ordinary-point coefficient recurrence is derived before any expansion,
and scalar heights, recurrence work, sequence digits, and serialized output
are admitted before the exact recurrence runs. It makes no claim about
convergence or an infinite sequence value. The coefficient-comparison rule
for an ordinary point is described in [DLMF §2.7](https://dlmf.nist.gov/2.7).
See [the finite-prefix contract](holonomic-differential-series-prefix.md)
for its recurrence and admission limits.

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

- [Differential Ore operator addition](ore-differential-addition.md)
- [Polynomial-coefficient Ore operators](ore-shift-polynomial-algebra.md)
- [Shift Ore operator powers](ore-shift-operator-powers.md)
- [Finite prefixes from polynomial recurrences](ore-shift-finite-recurrence.md)
- [Finite sequence prefixes for shift operators](ore-shift-sequence-prefix.md)

- [Exact cyclotomic polynomials](cyclotomic.md)
- [Elementary-symmetric polynomial families](elementary-symmetric.md)
- [Rational discrete antiderivatives](rational-discrete-antiderivative.md)
  compute the unique zero-based inverse of a selected-variable forward
  difference over `QQ`.
- [Monomial-ideal graded Betti profiles](monomial-ideal-graded-betti.md)
- [Exact rational Laurent-polynomial multiplication](rational-laurent-polynomials.md)
- [Exact root--critical-point distance profiles](root-critical-distance-profile.md)
