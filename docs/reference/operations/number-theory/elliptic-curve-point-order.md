# Finite-field elliptic-curve point order

`elliptic_curve.finite_field.point.order.compute` returns the exact additive
order of one curve-bound point. It first computes the curve cardinality by an
exact quadratic-character sum over the base field, then removes prime factors
from that cardinality while the corresponding reduced scalar still sends the
point to infinity.

The result includes the annihilating multiple `mP = O` and, for every prime
`r` dividing the reported order `m`, the nonidentity multiple `(m/r)P`. These
relations prove minimality. The identity point has order one and therefore an
empty prime-divisor witness list.

The operation admits base fields of order at most 4096. It bounds the exact
character-sum work, scalar multiplication steps, and serialized witnesses
before counting or scalar arithmetic. It does not enumerate or claim the
structure of the whole point group; group invariant factors and generators
need their own complete subgroup proof.

[Finite-field elliptic extension counts](elliptic-curve-extension-counts.md) ·
[Number theory operations](index.md) · [Operation reference](../index.md)
