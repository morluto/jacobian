# Petri-net target marking reachability

[Documentation home](../../index.md) · [Operation references](index.md)

`petri_net.marking.reachability.compute` searches for a firing sequence from
an initial marking to a selected target marking. `REACHABLE` returns a
transition sequence that can be supplied unchanged to
`petri_net.firing_sequence.replay.compute`; its replay reaches the requested
target. The operation does not claim the returned sequence is shortest.

`UNREACHABLE` is returned only when breadth-first exploration exhausts every
reachable marking without crossing an execution envelope. If the state limit
is reached, a successor exceeds the marking bound, or a witness would exceed
the replay operation's sequence bound, the result is `INCOMPLETE`. In
particular, a transition-count vector satisfying the state equation is not
evidence of an executable firing sequence.

The search retains visited markings and predecessor transitions rather than a
full edge list. Admission reuses the reachability state, token-cell, transition
expansion, and work bounds. The result records the explored-state count and
any envelope cut. A found witness remains a valid positive conclusion even if
another branch was cut; an exhausted search with any cut does not conclude
nonreachability.

The zero-step case is included: when the initial and target markings agree,
the result is `REACHABLE` with the empty sequence.
