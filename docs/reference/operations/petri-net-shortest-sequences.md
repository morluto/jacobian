# Petri-net all-shortest firing sequences

[Documentation home](../../index.md) · [Operation references](index.md)

`petri_net.reachability.shortest_sequences.compute` takes a complete finite
reachability graph and target marking. It returns the minimum firing count and
every transition-index sequence of that length. The result keeps the source net
and both endpoint markings. Distinct transition labels remain distinct when
they have the same source and target marking, as required by the labeled-edge
definition of a Petri-net reachability graph ([Zhu, Zhang, and Yang, 2017](https://journals.sagepub.com/doi/10.1177/1687814017700354)).

The operation validates every represented edge against the net firing rule,
checks that every state is reachable from state zero, and checks successor
closure. A truncated graph is rejected because it cannot establish either the
shortest distance or the complete family. A target absent from an accepted
complete graph has no sequence; a target equal to the initial marking has the
single empty sequence.

The complete family can be exponentially large. Before enumerating it, the
operation counts shortest paths in the distance-increasing subgraph and bounds
the serialized sequence family and total enumeration work. If either bound is
exceeded, it reports resource admission failure instead of returning a partial
family. Returned sequences fit the existing firing-sequence replay length
bound.
