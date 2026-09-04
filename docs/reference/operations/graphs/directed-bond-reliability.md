# Exact finite directed bond reliability

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`probability.digraph_bond_reliability.connection_probability.compute` computes
the exact probability that a directed open-arc path exists from one stated
source vertex to one stated target vertex. Each arc is independently open with
its supplied rational probability; traversal follows arc orientation.

The operation completely enumerates the bounded arc powerset and returns every
state's open arcs, exact mass, and directed reachability value. Its result
retains the canonical graph, terminal pair, and arc-probability map. The kernel
enumerates once and constructs the result without replay. Reversing the
terminals defines a different event.

The current exact envelope admits at most 12 arcs. Traversal materializes only the relevant vertices (the two terminals plus the endpoints
of open arcs), so executed work scales with arc count rather than with the
declared vertex count; declared vertex labels are bounded instead by the
interoperable JSON integer transport range. The arc limit bounds all 4096
states, the full ledger, and rational-product digit growth.
Edgeless directed graphs are admitted: with two distinct terminals the event
has exact probability zero and the ledger holds the single empty state.
Python-FLINT performs rational arithmetic, and an owner-local NetworkX
reachability call checks each state on its relevant vertices. Result parsing
checks structural consistency without independently certifying the ledger.

This is finite directed bond percolation only. It does not evaluate an
agent-authored event, condition terminals to be open, provide site or
hypergraph percolation, or decide a Bunkbed comparison.
