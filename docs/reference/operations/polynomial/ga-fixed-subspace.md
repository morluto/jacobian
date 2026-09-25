# Fixed subspaces of finite additive-group representations

`algebraic_group.ga.fixed_subspace.compute` takes a finite-dimensional
polynomial `G_a` subrepresentation and returns a deterministic basis of its
fixed vectors. Each output row also gives coordinates in the supplied ordered
basis, so the result retains its exact ambient representation context.

For a representation matrix `M(t)` over `QQ[t]`, the group law gives
`M(s+t) = M(s)M(t)`. Differentiating at `s=0` shows that a vector fixed by the
coefficient matrix `N = M'(0)` is fixed for every `t`; the converse follows by
differentiating a fixed vector's orbit at zero. The operation therefore
computes the exact rational kernel of `N`.

The input matrix is a caller-supplied claim. The operation recomputes the
representation matrix from the checked polynomial action and supplied basis,
rejecting a mismatched matrix before using it. This reconstruction uses the
bounded admission of `algebraic_group.ga.stable_subrepresentation.compute`;
fixed representatives are also admitted before construction. The dimension is
at most 32 and the returned polynomial support has at most 4096 terms.

This returns the fixed subspace inside one finite subrepresentation. It does
not find all finite subrepresentations or claim to generate the invariant ring
of the ambient polynomial algebra.
