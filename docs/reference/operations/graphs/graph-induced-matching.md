# Maximum induced matching

[Documentation home](../../../index.md) · [Graph operations](index.md) · [Tool surface](../../tools.md)

`graph.induced_matching.maximum.compute` computes a bounded exact maximum
induced matching of a finite simple graph. It returns the incumbent cardinality
and bounds, a canonical source-edge axis (`e0`, `e1`, …), the selected edge-ID
family, and the complete source subgraph on the selected endpoints.

Two selected source edges must be vertex-disjoint, and no other source edge may
join their endpoints. Internally the operation builds the square of the line
graph and invokes the existing bounded exact independence-number kernel; that
reduction is private and the result remains source-graph edges.

Source edges are sorted lexicographically before IDs are assigned. Among equal
cardinality optima, the selected ID family is lexicographically smallest. The
endpoint graph has sorted vertex and edge axes, independent of input row order.
The request admits at most 128 source edges and bounds conflict-pair and
intermediate graph-cell construction. An incomplete solver outcome is reported
with its feasible lower bound and safe upper bound; it is not an optimum claim.

This operation is distinct from ordinary maximum matching, induced bipartite
vertex-subgraph selection, strong edge coloring, weighted induced matching, and
minimum maximal matching.
