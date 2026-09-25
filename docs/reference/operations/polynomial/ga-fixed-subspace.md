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
fixed representatives are also admitted before construction. For the kernel,
the operation bounds the dimension-cubed work, each row's denominator lcm, and
the integerized matrix's Hadamard bound before exact elimination. The Hadamard
bound limits every minor to 10,000 decimal digits and therefore bounds the
unreduced Gauss-Jordan intermediates as well: each update multiplies two
minor-ratio entries and subtracts a third, so transient integer operands are
bounded by 30,002 digits. Kernel coordinates are bounded to 128 digits,
aggregate representative support to 4096 terms, and the combined result to the
exact output-byte envelope.

This returns the fixed subspace inside one finite subrepresentation. It does
not find all finite subrepresentations or claim to generate the invariant ring
of the ambient polynomial algebra.
