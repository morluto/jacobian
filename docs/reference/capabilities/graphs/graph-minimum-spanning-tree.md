# Exact weighted minimum spanning tree

[Documentation home](../../../index.md)

`graph.spanning_tree.minimum.compute` returns one deterministic
minimum-total-weight spanning tree of a bounded labelled simple graph with
exact rational edge weights. The producer remains `COMPUTED`.
When checker authority is installed,
`graph.spanning_tree.minimum.verify` can promote one exact submitted
`{input, candidate}` claim to `VERIFIED` after independent connectivity, tree, arithmetic, and
cycle-certificate replay.

## Input and outcome

The request contains at most 32 unique labelled vertices and at most 496
simple undirected edges. Each undirected endpoint pair occurs at most once and
has one reduced exact rational weight. Numerator and denominator components
are limited to 256 decimal digits and the denominator is positive.

For a connected nonempty graph the result contains:

- the canonical vertex order and connected-component partition;
- canonically oriented and sorted weighted tree edges;
- their exact rational total weight; and
- one fundamental-cycle check for every non-tree source edge.

A disconnected graph returns `NO_SPANNING_TREE` with its complete component
partition, no tree edges, and no total weight. The empty graph follows the same
convention. A singleton graph is connected and has the exact empty tree of
total weight zero.

The producer inserts edges in exact `(weight, left endpoint, right endpoint)`
order before calling NetworkX's maintained Kruskal implementation. This makes
the selected witness deterministic under request edge orientation and order.
Determinism chooses one optimum when ties exist; it does not claim uniqueness.

## Optimality certificate

For each non-tree edge \(e=uv\), the tree contains one unique \(u\)-to-\(v\)
path. The result records that path, the source weight \(w(e)\), and the maximum
tree-edge weight \(M_e\) on the path. The certificate requires

\[
w(e) \mathrel{\geq} M_e
\]

for every non-tree edge. If the inequality failed, replacing an edge attaining
\(M_e\) by \(e\) would produce a lighter spanning tree. Conversely, the
fundamental-cycle criterion for graphic matroid bases establishes that a
spanning tree satisfying every recorded inequality is minimum weight.

Certificate construction is untrusted. The producer cannot authorize its
checker, and a contract-valid certificate is not `VERIFIED` until the
operator-authorized replay succeeds.

## Independent replay

The checker imports neither NetworkX nor any Jacobian producer module. It uses
only bounded Python standard-library data structures and `fractions.Fraction`.
It independently parses the exact weighted graph, computes its connected
components, checks tree membership and spanning connectivity, recomputes the
total, reconstructs every unique tree path, and requires exact coverage of all
non-tree source edges.

| Obligation | Independent replay | Failure meaning |
| --- | --- | --- |
| Artifact binding | Recompute claim, semantics, candidate, lineage, witness-envelope, and payload bindings. | Reject this evidence; no mathematical conclusion. |
| Weighted graph validity | Reject malformed labels, loops, undeclared endpoints, parallel undirected edges, noncanonical rationals, and values outside the declared bounds. | Reject malformed or unsupported evidence. |
| Connectivity convention | Exhaust the supplied adjacency relation and compare the complete canonical component partition. | Reject a mismatched `EXACT` or `NO_SPANNING_TREE` outcome. |
| Spanning-tree feasibility | Require exactly \(n-1\) distinct source edges and reach every source vertex. | Reject a non-tree candidate. |
| Exact total | Sum the selected source weights with exact rational arithmetic. | Reject a forged total. |
| Certificate coverage | Require one canonical check for every and only every non-tree source edge. | Reject missing, duplicated, or substituted checks. |
| Cycle optimality | Independently reconstruct each tree path and its maximum edge weight, then check every non-improvement inequality. | Reject a feasible but nonminimum tree. |
| Authorization and runtime | Dispatch only the operator-authorized checker matching the exact schemas, semantics, evidence format, source digest, and provider runtime. | Unavailable, timeout, cancellation, or error remains non-conclusive. |

For a disconnected or empty graph, acceptance uses complete finite
connectivity replay. For a connected result, acceptance uses a checked
certificate. Every rejection returns `UNKNOWN`; it never identifies another
tree as optimal.

## Scope and remaining evidence

One invocation covers only the supplied finite weighted graph. It does not
certify a TSP tour, an approximation trace, a graph family, or an
all-orders theorem. Exact bounded TSP remains a separate deferred capability.

The public `metric-tsp-proof-repair` case motivated this fundamental primitive,
but its answer is public and no model run was performed for this change. A
held-out comparison across at least two independent weighted-graph families
remains required before making a model-performance claim.
