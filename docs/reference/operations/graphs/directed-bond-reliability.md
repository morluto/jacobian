# Exact finite directed bond reliability

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`probability.digraph_bond_reliability.connection_probability.compute` computes
the exact probability that a directed open-arc path exists from one stated
source vertex to one stated target vertex. Each arc is independently open with
its supplied rational probability; traversal follows arc orientation.

The operation completely enumerates the bounded arc powerset and returns every
state's open arcs, exact mass, and directed reachability value. Its result
retains the canonical graph, terminal pair, and arc-probability map, then
replays the complete enumeration during result validation. Reversing the
terminals is therefore a different event.

The current exact envelope admits at most 16 vertices and 12 arcs. The arc
limit bounds all 4096 states, the full ledger, rational-product digit growth,
and both the producer and replay passes. Python-FLINT performs producer
rational arithmetic; the standard-library `Fraction` replay checks those
values independently, while the existing directed-graph NetworkX operation
defines each state’s forward reachability predicate.

This is finite directed bond percolation only. It does not evaluate an
agent-authored event, condition terminals to be open, provide site or
hypergraph percolation, or decide a Bunkbed comparison.
