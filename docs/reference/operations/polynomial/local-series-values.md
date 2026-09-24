# Exact Puiseux window values

`TruncatedPuiseuxWindow` is the typed value carrier for a finite local prefix
with rational exponents. It stores a rational center, a rational retained lower
bound, a rational exclusive precision, and nonzero coefficient terms in
increasing exponent order. Missing exponent slots below the cutoff are known
zero; terms at or above the cutoff are unknown. An empty term list therefore
means zero through that precision, not an infinite zero series.

The explicit `ramification_index` is the least common denominator of the
retained bounds and term exponents. For example, the exponents `1/2` and `1/3`
require index `6`, while the cusp branch prefix `x^(3/2) + O(x^4)` requires
index `2`. Coefficients are exact rationals, and the value serializes with its
center, exponent lattice, and cutoff intact.

`TruncatedLaurentWindow` is the integral-exponent representation for Laurent
arithmetic. Its `place` is `FINITE` for parameter `t = x - center`, or
`INFINITY` for reciprocal parameter `t = 1/x` (with center zero). Arithmetic
preserves this parent and rejects mixing different places. Exact univariate
rational-function producers exist for rational finite centers and infinity.
Exact conversions connect the power-series carrier `TruncatedSeries` with the
integral Laurent subtype. Conversion from power series preserves the order and
variable while normalizing known leading zeros. Conversion back requires the
finite center-zero parent and precision supported by #1713. A nonzero known
negative coefficient returns `HAS_NEGATIVE_EXPONENTS` with the first such
exponent; it does not discard the principal part. Local Newton polygons and
Newton–Puiseux branch computation remain outside this slice.

## Bounded arithmetic

The operations `local_series.puiseux.add.compute`,
`local_series.puiseux.subtract.compute`, and
`local_series.puiseux.multiply.compute` act on windows with the same variable
and rational center. They combine the inputs on the least common ramification
lattice, capped at index 256, and retain only nonzero terms. Addition and
subtraction stop at the smaller input cutoff.

Multiplication does not treat either omitted tail as zero. If the retained
prefixes have valuations `v_f`, `v_g` and cutoffs `P_f`, `P_g`, its safe
exclusive cutoff is
`min(P_f + v_g, P_g + v_f)`. An empty prefix has valuation lower bound equal
to its cutoff. The product admits the input term-pair work, output support,
exponent window, and worst-case exact coefficient growth before forming the
Cauchy product. Every operation also admits a conservative serialized-result
size estimate below the canonical 10 MiB output limit. The estimate includes
the maximum exact numerator and denominator sizes for every coefficient,
bounded exponent fields, center and window bounds, and JSON structure. These
operations return finite exact prefixes; they make no claim about the unknown
tails.

`local_series.puiseux.derivative.compute` applies
`d(c*t^q)/dt = q*c*t^(q-1)` termwise, drops the constant term, and lowers
both retained exponent bounds by one. An input known modulo `O(t^P)` yields a
derivative known modulo `O(t^(P-1))`. An empty retained prefix remains an
empty prefix and does not assert that the underlying series vanishes. The
implementation admits term work, coefficient growth, the shifted exponent
window, and serialized output before constructing the result.

`local_series.puiseux.inverse.compute` requires a retained nonzero leading
term. If its exponent is `v` and the source cutoff is `P`, the returned inverse
has valuation `-v` and cutoff `P-2v`: writing the source as
`a*t^v*(1+u(t))`, the known unit prefix determines its reciprocal only modulo
`O(t^(P-v))`, which shifts by `t^(-v)`. The common least-ramification lattice
indexes an exact integer recurrence for the unit reciprocal. An empty finite
prefix is rejected as undetermined because its omitted tail may have a
nonzero leading term; this does not conclude that the underlying series is
noninvertible. Before recurrence expansion, the operation admits lattice span,
output support and bytes, term work, normalized unit denominators, recurrence
coefficient growth, and all generated exact coefficients.

`local_series.puiseux.residue.compute` returns the exact coefficient of
`t^-1`, source-bound to the input window. It accepts the result only when the
known exponent interval contains `-1`; a window above or below that exponent
does not establish a zero residue because the omitted portion is unknown.
When `-1` lies inside the interval but has no retained sparse term, its
coefficient is known to be zero.
