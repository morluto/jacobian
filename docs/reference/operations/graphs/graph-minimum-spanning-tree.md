# Exact weighted minimum spanning tree

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.spanning_tree.minimum.compute` accepts a bounded labelled simple graph
with exact rational edge weights. It returns one deterministic minimum-total-
weight spanning tree, its total, connected components, and the finite
non-improvement checks for the selected edges.

An empty or disconnected graph returns the operation's typed
`NO_SPANNING_TREE` outcome. The graph and returned tree remain ordinary inline
values; no record, replay job, or evidence store is involved.
