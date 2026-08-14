# Maximum matching

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.invariant.maximum_matching.compute` computes an exact
maximum-cardinality matching of a bounded simple graph and returns its matching
edges together with the domain evidence carried by that result. It is a direct
typed computation: the graph and matching are supplied and returned inline.

`graph.matching.maximal.minimum.compute` is a different operation for the
minimum size of a maximal matching. Neither operation creates a graph record or
requires a follow-up checker call.
