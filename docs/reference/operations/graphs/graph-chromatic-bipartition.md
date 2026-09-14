# Chromatic bipartition feasibility

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.chromatic_bipartition.find` decides the finite `(s,t)`-splittability
predicate for one canonical simple graph. It searches unordered nonempty proper
vertex bipartitions. A `SPLIT` result retains both sides in source-vertex order
and the exact chromatic number of each induced subgraph. A `NO_SPLIT` result is
returned only after the admitted partition search is complete (or after a
sound shortcut such as `s + t > |V|`); it carries no witness.

If the shared admitted wall-clock envelope or an inner exact coloring decision
is exhausted, the kernel raises an execution timeout. That is an operational
failure, never a mathematical `UNKNOWN` or `NO_SPLIT` conclusion. The result
type admits only `SPLIT` and `NO_SPLIT`; an incomplete but healthy worker
frame that claims `UNKNOWN` is an execution failure. If the killable worker is
terminated before its result frame arrives, the parent reports
`OperationExecutionTimeoutError`.

The operation is bounded independently of the graph carrier: witness-returning
branches admit at most 256 source vertices. Cheap exact `NO_SPLIT` shortcuts
with no witness, including an edgeless graph when `s > 1` or `t > 1` and the
threshold-sum obstruction `s + t > |V|`, keep that exact negative result above
the witness-axis cap and do not construct a hypothetical split envelope.
Retained-label admission charges repeated witness axes only when a `SPLIT`
result is possible. The operation also admits at most one million retained
label characters (including the source graph and, when a witness is possible,
repeated witness axes), and
two million charged units of partition reconstruction plus exact coloring work. It uses the maintained
NetworkX graph adapter and Z3 k-colorability kernel privately; backend identity
is not part of the result contract. Thresholds are directional, so a returned
`side_a` satisfies `s` and `side_b` satisfies `t` even when the underlying
unordered partition is considered in the opposite orientation.

For example, `K4` is `(2,2)`-splittable by `{a,b}` and `{c,d}`, while `K3` is
not `(2,2)`-splittable because one side of every bipartition has at most one
vertex.
