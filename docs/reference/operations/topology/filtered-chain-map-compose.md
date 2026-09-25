# Compose filtered chain maps

`homological.filtered_chain_map.compose.compute` composes two degreewise
matrices `C -> D -> E` into the exact map `C -> E`. It consumes two results of
the filtered-map operation directly, including after JSON serialization. The
coefficient field and degree axes must agree, the middle complexes must be
identical, and the two middle filtrations must define the same subspaces (their
spanning-vector lists may differ). The operation rechecks each supplied map's
chain and filtration relations; caller-supplied status fields do not establish
them.

The result retains the first source and last target, both filtration axes, and
the degreewise product matrices. It can be supplied unchanged as a map to
another filtered-map operation after serialization. Composition uses the
standard degreewise composition of chain maps (see the chain-complex category
construction in [Chapter 3, Homological Algebra](https://agag-lassueur.math.rptu.de/~lassueur/en/teaching/COHOMSS18/CGSS18/Kap3.pdf)).

Before multiplying, the operation bounds map cells, exact multiplication work,
coefficient growth, and serialized output size. Input complexes and filtrations
use the same field and size limits as the existing filtered-map operation.
