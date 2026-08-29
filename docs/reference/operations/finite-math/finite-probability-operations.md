# Finite probability operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Finite distributions are canonical bounded rational values. Jacobian provides
direct operations for conditioning, convolution, i.i.d. convolution powers,
pushforward, and raw moments:

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
