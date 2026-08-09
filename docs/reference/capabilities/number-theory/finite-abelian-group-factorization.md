# Finite abelian group exact factorization

`finite_abelian_group.exact_factorization.compute` works in a bounded product
of cyclic groups `Z/n1 x ... x Z/nr`. It normalizes two supplied lists of
integer vectors and exhaustively counts representations of every group element
as `a+b` with `a` from the left factor and `b` from the right factor.

The result contains the normalized factors, the complete representation-count
histogram, an exact-factorization decision, and the first missing and duplicate
representation witnesses when the decision is false. A complete coset
transversal is the special case where the right factor contains only zero.

The current contract supports rank at most six, group order at most 4,096,
factor sizes at most 256, and a factor Cartesian product no larger than the
maximum supported group order (4,096). Coordinates may be noncanonical integers and are reduced by their
corresponding cyclic moduli. Each factor must remain a set after normalization;
congruent duplicate entries are rejected before computation.

The producer returns `COMPUTED` evidence. The operator-authorized companion
`finite_abelian_group.exact_factorization.verify` independently normalizes the
payload and replays every group sum, binding the accepted record to the exact
group presentation, factors, histogram, decision, and witnesses.

This capability verifies only the supplied finite factorization. It does not
decide infinite lattice tiling, periodicity, or orbit-closure properties.
