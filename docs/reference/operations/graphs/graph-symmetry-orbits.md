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
value. The kernel computes the vertex and edge orbits once; result parsing
checks canonical partition shape and source references without recomputing
the generated subgroup orbits.
The result echoes its complete source plus the derived partitions. Request
admission bounds graph, generator, vertex-orbit, and edge-orbit cardinalities;
native execution does not inherit a JSON response-byte ceiling.

The operation has no global symmetry registry, stored graph source, or replay
product.
