# Declared graph-symmetry orbits

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`graph.symmetry.generator_orbits.compute` validates the submitted
color-preserving automorphism generators of a bounded graph and computes the
complete vertex and edge orbit partitions of their generated subgroup.

The generators, graph, and orbit partition are typed inline values. Each
generator declares a total vertex permutation as `(vertex, image)` pairs
covering every declared vertex exactly once in the graph's declared vertex
order. Generator identifiers and declared color labels must already be
normalized to Unicode NFC; decomposed forms are rejected by request
validation, and this requirement is published in the operation's request
schema.

The result retains the complete declared source action - canonical graph,
generator mappings, and declared colors - through the domain-owned request
value, and its validation replays the exact vertex and edge orbits of that
declared generated subgroup against the returned partitions. Because the
result echoes its complete source plus the derived partitions, request
validation also applies one aggregate retained-result bound: it measures the
complete canonical serialization before execution and rejects any request
whose result would exceed Jacobian's canonical output limit, even when every
field-level bound is satisfied. This aggregate envelope is published in the
request schema alongside the per-field bounds.

The operation has no global symmetry registry, stored graph source, or replay
product.
