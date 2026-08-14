# Finite probability operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Finite distributions are canonical bounded rational values. Jacobian provides
direct operations for conditioning, convolution, event probability,
pushforward, and raw moments:

- `probability.finite_distribution.condition.compute`
- `probability.finite_distribution.convolution.compute`
- `probability.finite_distribution.event_probability.compute`
- `probability.finite_distribution.pushforward.compute`
- `probability.finite_distribution.raw_moment.compute`

Each request includes the distribution and any event or map it needs. Each
result is returned inline; no distribution or calculation is retained.
