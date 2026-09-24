# Normalized chains of a finite simplicial set

`topology.simplicial_set.normalized_chains.compute` returns the normalized
chain complex of a complete table based finite simplicial set prefix
`X_0,...,X_N`. In degree `n`, its ordered basis consists of the source labels
whose simplices are not in the image of any visible degeneracy map
`s_i : X_(n-1) -> X_n`. The differential is the alternating face sum, with
terms landing in the degenerate subspace set to zero in the quotient.

The result retains the source prefix, the exact nondegenerate label tuple for
each degree, and the shared `ChainComplexValue`. Its chain axes therefore
compose directly with chain-complex operations while preserving the labels
needed to interpret matrix rows and columns. The operation rechecks the
caller supplied simplicial identities, then checks that the normalized
differential squares to zero before returning. It retains the formal top group
in degree `N`; it does not infer a differential from a missing degree `N+1`.

The matrix-cell and aggregate output-byte bounds are checked from the source
degree sizes and serialized axes before identity replay or matrix construction.
Coefficients are integral. Use the unnormalized
chain operation when every simplex, including degenerate ones, should remain
in the basis.
