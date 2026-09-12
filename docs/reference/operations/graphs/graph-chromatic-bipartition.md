# Chromatic bipartition feasibility

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.chromatic_bipartition.find` decides the finite `(s,t)`-splittability
predicate for one canonical simple graph. It searches unordered nonempty proper
vertex bipartitions. A `SPLIT` result retains both sides in source-vertex order
and the exact chromatic number of each induced subgraph. A `NO_SPLIT` result is
returned only after the admitted partition search is complete (or after a
sound shortcut such as `s + t > |V|`); it carries no witness.

If the shared admitted wall-clock envelope or an inner exact coloring decision
is exhausted, the result is `UNKNOWN`, never a negative conclusion. `UNKNOWN`
retains the source graph, thresholds, and a conservative count of partition
candidates reported as checked by the completed kernel. If the killable worker
is terminated before its result frame arrives, that count is zero because the
parent cannot recover child progress; no partial witness or chromatic claim is
retained.

The operation is bounded independently of the graph carrier: it admits at most
256 source vertices, one million retained label characters (including the
source graph and repeated witness axes), and two million charged units of
partition reconstruction plus exact coloring work. It uses the maintained
NetworkX graph adapter and Z3 k-colorability kernel privately; backend identity
is not part of the result contract. Thresholds are directional, so a returned
`side_a` satisfies `s` and `side_b` satisfies `t` even when the underlying
unordered partition is considered in the opposite orientation.

For example, `K4` is `(2,2)`-splittable by `{a,b}` and `{c,d}`, while `K3` is
not `(2,2)`-splittable because one side of every bipartition has at most one
vertex.
