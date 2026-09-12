# Finite probability operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Finite distributions are canonical bounded rational values. Jacobian provides
direct operations for conditioning, convolution, i.i.d. convolution powers,
pushforward, and raw moments:

- `probability.compound_poisson.cumulant_prefix.compute`
- `probability.finite_distribution.condition.compute`
- `probability.finite_distribution.convolution.compute`
- `probability.finite_distribution.convolution_power.compute`
- `probability.finite_distribution.convolution_peak.compute`
- `probability.finite_distribution.pushforward.compute`
- `probability.finite_distribution.raw_moment.compute`

The power operations translate and scale every positive-mass rational support
onto its minimal integer lattice, clear the probability denominators, and use
FLINT integer-polynomial multiplication. Request admission separately bounds
the lattice span, every dense coefficient product in binary exponentiation,
coefficient height, and the complete canonical result. The power result returns
the entire exact distribution. The peak result instead returns the exact
largest mass and every atom attaining it, bound to the retained source law and
exponent; it does not emit the binary convolution contribution ledger.

Each request includes the distribution and any event or map it needs. Each
result is returned inline; no distribution or calculation is retained.
The probability of an explicit event is a direct sum of selected rational
masses and is intentionally left to ordinary Python rather than occupying a
separate public discovery slot.

## Compound-Poisson cumulant prefixes

`probability.compound_poisson.cumulant_prefix.compute` accepts a nonnegative
exact rational intensity `lambda`, one normalized finite rational jump law, and
a maximum order. It returns the rows

\[
  (n, m_n, \kappa_n) = (n, E[J^n], \lambda E[J^n])
  \quad\text{for } 1 \le n \le \text{max\_order}.
\]

The rows are a finite exact invariant of the compound-Poisson law, not a PMF
prefix. A compound-Poisson sum generally has infinite support and its masses
need not be rational, so this operation does not truncate or materialize that
law. `max_order=0` returns an empty prefix. Signed rational jump values,
deterministic jumps, zero intensity, and Bernoulli jumps are all supported.

The operation admits support size, moment work, input rational height,
intermediate exact rational growth, and every returned cumulant before
constructing the result. Powers for all requested orders share one ladder, so
admission and execution do not recompute earlier moments. The returned source
retains the intensity and jump parent for exact downstream interpretation.
