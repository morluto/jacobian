# Maximum matching

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`graph.invariant.maximum_matching.compute` computes an exact
maximum-cardinality matching of a bounded simple graph and returns its matching
edges together with the domain evidence carried by that result. It is a direct
typed computation: the graph and matching are supplied and returned inline.
It creates no graph record and requires no follow-up checker call.

`graph.matching.maximal.minimum.compute` is a separate published operation
that computes the minimum size of a maximal matching. It differs from maximum
cardinality matching above: the former minimizes over maximal matchings, while
the latter finds a matching of maximum cardinality.
