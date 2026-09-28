# Petri-net terminal SCC profiles

`petri_net.reachability.terminal_scc_profile.compute` takes one bounded
reachability graph and returns the state-index sets of its sink strongly
connected components. The input graph is checked for unique reachable states,
legal transition-labelled edges, and closure when it declares itself complete.

When the source graph is untruncated, these are the exact terminal SCCs of the
finite reachable state graph. A singleton is terminal whether or not it has a
self-loop; terminal does not mean deadlocked. When the graph is truncated, the
result describes sink SCCs only in the represented partial graph. It makes no
claim about omitted successors, full-net recurrence, or boundedness.

The definition follows finite Petri reachability analysis: an SCC is terminal
when no vertex in it has a successor in another SCC. Liveness conclusions in
the literature additionally require the full reachability graph; this
operation returns only the structural SCC profile ([Boukala and Petrucci,
“Towards Distributed Verification of Petri Nets Properties”](https://citeseerx.ist.psu.edu/document?doi=c625a801b8b047fc3ec68cb4ffdd5c1ebb5eb52e&repid=rep1&type=pdf)).

The operation admits state/edge work, canonical ordering work for the returned
component tuples, and worst-case output bytes before it builds adjacency or
computes components. Its iterative SCC traversal supports large state counts
without Python recursion depth dependence.
