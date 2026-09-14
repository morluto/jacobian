# Bounded regular plane-curve arclength

`real_algebraic.plane_curve.arclength.enclose` returns a source-bound rational
interval containing

\[
  \mathcal H^1(\{(x,y):F(x,y)=0\}\cap B),
\]

for one exact rational polynomial and one closed rational box.  It is an
enclosure, not a decimal approximation or an extremal theorem.

## First admitted envelope

The initial kernel admits a regular axis-aligned ellipse

\[
 A(x-h)^2+C(y-k)^2=R,\qquad A,C,R>0,
\]

written as a rational polynomial with no `xy` term.  Exact nonzero constants
are admitted as the empty curve.  Singular quadratics, the zero polynomial,
non-diagonal quadratic parts, higher-degree sources, and unresolved irrational
boundary intersections are outside this envelope.  A repeated factor is
therefore never counted with multiplicity.

The result disposition is one of:

- `ENCLOSED`, with rational `lower`, `upper`, and a finite source-bound
  parameter-cell partition;
- `EMPTY`, with exact zero length;
- `SINGULAR_CASE_UNSUPPORTED`, for a degenerate level set or nontransverse
  boundary; or
- `UNKNOWN`, for unresolved topology, exhausted validated refinement, backend
  absence, or deadline expiry.  `UNKNOWN` carries no length claim.

The box and polynomial use the same complete ordered two-variable axis.  The
positive rational `target_width` is charged semantically; accepted enclosures
have `upper - lower <= target_width`.  Degree, terms, coefficient and box
endpoint height, precision, parameter cells, quadrature leaves, result size,
and one shared wall deadline are bounded before Arb work begins.

## Certified computation

The regular ellipse is represented by a rational half-angle parameter.  Finite
parameter cells and the reciprocal chart at infinity cover the locus once.
Every cell contribution is produced by the existing validated definite-integral
operation using pinned python-flint Arb balls and outward-rounded dyadic
endpoints, then summed exactly as rationals.  A quadrature that does not meet
the requested width remains `UNKNOWN`; no solver success flag or plot is a
mathematical result.

The maintained backend is python-flint 0.9.0, whose Arb values track rigorous
midpoint-radius balls and whose calculus interface documents interval-based
integration with explicit no-convergence outcomes.  See the
[python-flint documentation](https://python-flint.readthedocs.io/en/latest/)
and [FLINT's Arb calculus contract](https://flintlib.org/doc/arb_calc.html).

## Example

For `F=x²+y²-1`, `B=[-2,2]²`, and `target_width=1/100`, the returned interval
contains `2π` and has width at most `1/100`.  It is intentionally not forced
to serialize `π` as an exact rational.

The operation does not claim support for singular lemniscates such as
`(x²+y²-1)²`; a future reduced-locus/branch-topology extension needs a
separate admitted contract to avoid multiplicity errors.
