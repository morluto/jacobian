# Petri-net transition liveness

`petri_net.transition_liveness.profile.compute` classifies each transition of a
Petri net as live, not live, or unknown, given a complete finite reachability
graph.

A transition is **live** exactly when, from every reachable marking, some
continuation can fire it. **Not live** means a reachable marking is represented
from which no continuation fires the transition. The witness state is the
index of such a marking in the supplied graph.

## Completeness

A transition is live only when the conclusion follows from the whole represented
state set. When the supplied graph is truncated, the graph does not contain every
reachable marking, so the result reports `UNKNOWN` for every transition — including
observed self-loops. A `LIVE` or `NOT_LIVE` status is never reported from a
partial graph.

## Bounds

The result retains the source graph and one entry per transition. Admission counts
those retained entries as cells against the transition-liveness output cell bound;
it does not estimate an encoded size. Scalar token magnitudes stay exact, so a
larger marking does not by itself admit a larger profile.

## Relation to the terminal SCC profile

Both operations read the same complete finite reachability graph. The terminal
SCC profile reports strongly connected components and their cycle structure;
this operation reports, per transition, whether it can still be fired from every
reachable marking. Neither establishes that a reported state is reachable in the
original net beyond what the supplied graph already asserts.
