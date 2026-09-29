# Maximum matching

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`graph.invariant.maximum_matching.compute` computes an exact
maximum-cardinality matching of a bounded simple graph and returns its matching
edges together with the domain evidence carried by that result. It is a direct
typed computation: the graph and matching are supplied and returned inline.
It creates no graph record and requires no follow-up checker call.

A solver-internal `graph.matching.maximal.minimum.compute` (minimum size of a
maximal matching) exists in the finite-optimization backend but is not
published to the catalog; do not look for it via `math.find`.
