# Finite relational homomorphism maps

[Operation references](index.md) · [Tool surface](../tools.md)

`RelationalCarrierMap` binds a complete map to its exact source and target
structures over the same ordered ranked signature. Mapping position `i` is the
target label assigned to source label `i`. Injectivity, surjectivity, and the
image are derived from that map.

`RelationalHomomorphism` is the corresponding map after relation preservation
has been established: every tuple in each source relation maps into the target
relation with the same symbol. Decoding checks the endpoint structures and map
shape; it does not replay preservation. A consumer that relies on a supplied
homomorphism claim checks it at its operation boundary.

`relational_homomorphism.identity.compute` returns the canonical identity map
on one structure, including for an empty carrier. The result is directly
usable as a composition input after serialization.

`relational_homomorphism.compose.compute` takes `first: A -> B` and
`second: B -> C`, checks that the intermediate structures are exactly equal,
then checks both supplied homomorphism claims over all source relation tuples.
It returns `second ∘ first`. Each structure carrier has at most 64 labels; the
two preservation replays are jointly bounded by 32,768 tuple visits. The
composite map itself is linear in the source carrier size.

Nullary relations are checked as truth values. A true nullary source relation
must map to a true target nullary relation; false source relations impose no
preservation condition. An empty source carrier has the unique empty identity
map.
