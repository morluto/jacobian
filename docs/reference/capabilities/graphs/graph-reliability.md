# Small exact graph reliability

[Documentation home](../../../index.md)

- Operation: `probability.graph_reliability.connection_probability.compute`
- Verification: `probability.graph_reliability.connection_probability.verify`
- Producer: pinned Python-FLINT exact rational arithmetic
- Checker: isolated standard-library `Fraction` enumeration and graph traversal

The operation computes one atomic outcome: the exact probability that two
explicit terminals are connected when every edge of one finite simple
undirected graph is independently open with its declared rational probability.

## Bounded contract

The request uses the existing canonical `SimpleUndirectedGraph`, at most 16
vertices and 12 edges, one rational probability in \([0,1]\) for every edge in
the graph's exact order, and two distinct declared terminals. Full request
validation precedes computation and artifact writes.

The producer exhausts all \(2^{|E|}\) edge subsets. Every state records its
canonical index, open edges, exact product-measure mass, and terminal
connectivity. The result reports `visited_states`, `COMPLETE_EDGE_SUBSETS`,
`COMPLETE`, `truncated=false`, and `EXHAUSTED` separately from execution and
assurance. It therefore has no prefix, timeout, or failed-witness semantics.

The producer remains `COMPUTED`. The operator-authorized checker independently
parses the graph and probabilities, enumerates the entire powerset with
standard-library `Fraction`, recomputes connectivity by graph traversal, and
binds every state and the final sum to the stored request and result.

## Excluded claims

This operation does not construct bunkbed graphs, compare two events, exploit
symmetry, emit a reliability polynomial, prescribe a search, or support more
than 12 edges. In particular, the explicit 7,222-vertex counterexample in
[The bunkbed conjecture is false](https://arxiv.org/abs/2410.02545) is
answer-visible motivation, not an acceptance target. Scaling that case requires
separate compressed graph and symmetry artifacts; raising a JSON limit would
not establish a complete calculation.

The separately verified
[declared graph-symmetry orbit](graph-symmetry-orbits.md) operation exposes
vertex and edge compression metadata under supplied generators. It does not
lift that action to edge states or establish a reliability identity.
