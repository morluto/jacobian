# Cellular sheaf cohomology maps

`cellular_sheaf.morphism.cohomology_map.compute` derives the linear maps on
cellular sheaf cohomology induced by a natural stalk morphism. It first
re-establishes naturality, builds the corresponding cochain map, and computes
both cohomology values from their source sheaf diagrams. It does not trust
serialized naturality flags or submitted cocycle claims.

For each degree, the operation applies the cochain map to the source's returned
cocycle representatives. It expresses those images in the target's direct sum
of coboundaries and cohomology representatives, then returns only the
cohomology coordinates. Thus each output matrix has target Betti number rows
and source Betti number columns. The result retains the inducing morphism and
both ordered cohomology group collections, whose representatives define the
matrix axes. The cochain map is recomputed internally and its cochain-level
matrices are not duplicated in the result.

This is the finite exact linear-algebra form of cellular-sheaf functoriality:
a sheaf morphism induces cochain maps commuting with coboundaries, hence maps
on cohomology ([Hansen and Ghrist, *Toward a Spectral Theory of Cellular
Sheaves*, §2](https://arxiv.org/abs/1808.01513)).

The exact field and source complex must agree across the sheaf morphism. The
quotient reduction is admitted against scalar-work, intermediate rational
growth, result-cell, and serialized-size bounds before its quotient matrices
are expanded. This operation supplies functorial target coordinates for later
kernel, image, and cokernel computations. It does not itself compute those
subspaces.

For a constant rank-one sheaf on a simplicial circle, scalar multiplication by
`c` induces multiplication by `c` on both `H^0` and `H^1`.
