# Normalized chain map induced by a finite simplicial map

`topology.simplicial_set.map.induced_chain_map.compute` constructs the chain
map on normalized integral chains induced by a finite truncated simplicial map.
It returns a reusable `ChainMapValue` whose source and target are the exact
normalized `ChainComplexValue`s. Optional basis labels retain the ordered
nondegenerate simplices on each endpoint axis.

For each degree, a nondegenerate source simplex maps to its image when that
image is nondegenerate, and maps to zero when the image is degenerate. This is
the induced map on the quotient of unnormalized chains by degenerate chains.
The operation checks that the resulting matrices commute with the normalized
boundaries before returning. The standard normalized Moore complex is functorial
in simplicial objects; see [Mathlib's Dold–Kan reference](https://leanprover-community.github.io/mathlib4_docs/Mathlib/AlgebraicTopology/DoldKan/Equivalence.html).

The operation rechecks all visible simplicial identities of both map carriers
and every face and degeneracy naturality square. Input carriers are bounded to
degree 4 and 96 simplices. Admission bounds the dense map-matrix cells, both
normalized boundary matrices, chain-map multiplication work, and output cells
before constructing the matrices. The value describes only the common finite
prefix; it makes no assertion above that degree.

The returned chain map can be passed unchanged to
`chain_complex.verify_chain_map.compute` or
`chain_complex.mapping_cone.compute`. Each consumer checks the chain-map
relation at its own admitted boundary.
