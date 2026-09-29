# Delta-matroids from looped graphs

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

The `graph.looped_adjacency_delta_matroid.compute` operation constructs a delta-matroid
from a graph that may carry loops, by way of its binary adjacency matrix.

The operation forms the symmetric GF(2) adjacency matrix on the graph's ordered
vertex axis, treating each loop as a diagonal entry rather than discarding it.
It then returns the complete family of vertex subsets whose induced principal
submatrix is nonsingular over GF(2). That family is the delta-matroid: the
feasible sets of a binary principal-minor construction.

The ground axis is the graph's vertex axis, so every returned feasible subset is
indexed by the same positions the caller supplied. A loop contributes a `1` on
the diagonal, which is what distinguishes this construction from the loop-free
adjacency case: the looped matrix is not the incidence matrix of a graph, and its
principal minors carry information the loop-free version cannot express.

The exact binary principal-minor work bound admits at most 8 vertices. The bound
is on the admitted call, not on the size of the returned family, which is
materialized in full.

The operation returns the source graph together with the feasible family; it does
not compute a presentation, a basis exchange, or a duality for the result. For
the general binary principal-minor surface and the related delta-matroid
operations, see [finite-delta-matroids.md](../finite-math/finite-delta-matroids.md).
