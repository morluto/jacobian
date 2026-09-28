# Petri-net reachable dead markings

[Documentation home](../../index.md) · [Operation references](index.md)

`petri_net.reachable_dead_markings.compute` explores the bounded reachable
state space and returns the discovered markings at which no transition is
enabled. Deadness is checked against the net's input arcs at each marking;
the operation does not infer deadness from a missing edge in a truncated
reachability graph.

The result's `truncated` flag is true if the state or token bound cut an
enabled firing. In that case, `dead_markings` remains the exact set of dead
markings among the discovered states, but it may omit dead markings whose
paths cross the cut. When `truncated` is false, it is the complete set of
reachable dead markings.

Admission bounds the reachability state, place, transition and work products,
then bounds the worst-case serialized marking list before exploration. The
operation shares the bounded reachability kernel and does not return or replay
a second graph artifact.
