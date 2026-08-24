# Declared graph-symmetry orbits

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.symmetry.generator_orbits.compute` validates the submitted
color-preserving automorphism generators of a bounded graph and computes the
complete vertex and edge orbit partitions of their generated subgroup.

The generators, graph, and orbit partition are typed inline values. The
result retains the complete declared source action - canonical graph,
generator mappings, and declared colors - through the domain-owned request
value, and its validation replays the exact vertex and edge orbits of that
declared generated subgroup against the returned partitions. The
operation has no global symmetry registry, stored graph source, or replay
product.
