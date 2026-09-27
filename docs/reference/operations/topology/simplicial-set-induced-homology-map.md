# Induced map on normalized integral homology

`topology.simplicial_set.map.induced_homology.compute` computes the homology
map induced by a degreewise simplicial map between finite truncated
simplicial sets.

For a prefix through degree `N`, the result covers degrees `0` through `N-1`.
The prefix contains the incoming differential needed in those degrees; it does
not determine `H_N`, so the top degree is omitted. Both endpoints retain their
normalized chain complexes and exact cycle representatives.

In each degree, the result gives the image of every retained free and torsion
generator. Coordinates use the exact decomposition

```text
H_n = Z^r ⊕ Z/d_1 ⊕ ... ⊕ Z/d_t,
```

with free coordinates in `Z` and torsion coordinates as residues in
`0..d_i-1`. Zero-rank groups are retained with empty generator-image rows.
The value checks source and target axes and verifies that each torsion
generator's order annihilates its image.

The operation rechecks simplicial-map naturality, constructs the induced
normalized chain map, checks that it commutes with the boundary, and checks
the source and target cycle and torsion bounding-chain identities on the
retained representatives. `compose_simplicial_homology_maps` composes two
source-bound results after requiring equality of the intermediate homology
values.

The exact integral homology kernel admits each endpoint using its rank,
matrix-cell, coefficient-digit, Smith-work, and output limits. The induced-map
matrices are bounded by those endpoint ranks and the finite-prefix degree
limit. The map is integral because normalized simplicial chains use the
integral coefficient ring.
