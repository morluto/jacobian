# Petri-net reachability token profile

[Documentation home](../../index.md) · [Operation references](index.md)

`petri_net.reachability.token_profile.compute` reports the minimum and maximum
token count for each place, plus the minimum and maximum total token count, over
the markings represented in a `ReachabilityResult`. Each extremum includes the
first state index attaining it in the source graph's canonical BFS order.

For a complete graph these extrema describe the entire finite reachable set.
For a truncated graph they describe only the represented reachable states and
are marked `OBSERVED_PREFIX`; they do not establish a bound on omitted states
or on the net's full reachability set. This follows the standard reachability
graph interpretation of vertices as reachable markings; see Murata,
[“Petri Nets: Properties, Analysis, and Applications,” §3](https://www.dsc.ufcg.edu.br/~abrantes/CursosAnteriores/MVSRP/murata89.pdf).

The operation checks the supplied graph's state and edge semantics, including
full successor closure when it claims to be complete. Its work and serialized
result are bounded before the graph is copied and its firing edges are replayed.
