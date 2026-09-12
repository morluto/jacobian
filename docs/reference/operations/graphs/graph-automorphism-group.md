# Full graph automorphism group

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.symmetry.automorphism_group.compute` computes the complete
color-preserving automorphism group of one finite simple undirected graph with
at most 64 vertices. The graph's edge carrier remains bounded by the shared
simple-graph value; the operation's permutation action is limited to degree 64.
Vertex and edge colors are preserved exactly. The returned `group` is a
canonical `PermutationGroup` acting on the lexicographically sorted vertex
axis, so it can be passed unchanged to `group.order.compute` and
`group.orbit.compute`.

The result also retains source-label generators and complete vertex and edge
orbits. Pass the retained graph and generators to
`graph.symmetry.generator_orbits.compute` to independently replay the orbit
partition. Generator identifiers and ordering are deterministic. A trivial
group uses the identity permutation internally; the empty graph uses the
degree-zero identity action and has order one.

Common high-symmetry families such as complete, empty, path, cycle, and
repeated clique graphs use compact presentations. Other graphs use a bounded
NetworkX VF2 automorphism search. Admission accounts for the vertex/edge
carrier, candidate search space, edge-scan work, generator rows, and orbit
partitions before search begins. A request outside that envelope is rejected
as a resource admission failure; an interrupted or malformed backend never
becomes a claimed subgroup.

Graph isomorphism, canonical labeling, arbitrary relational structures, and
classification up to isomorphism remain separate operations.
