# Affine semigroup group lattice

`affine_semigroup.group_lattice.compute` returns the integer subgroup generated
by the labelled columns of a bounded affine configuration
`A ∈ ZZ^(d×n)`. Its `IntegerLattice` result is the canonical row-Hermite basis
of those columns, stored in the configuration's ambient coordinates. The
returned value retains the full source configuration, including redundant and
zero generators. Two integral coordinate matrices exhibit both inclusions
between the source generators and the returned basis, so serialized values
remain bound to exactly the claimed generated subgroup.

The subgroup is `ZA = {A u : u ∈ ZZ^n}`. It is distinct from the nonnegative
affine semigroup `NA`; the operation makes no positivity, cone, or membership
claim. Its canonical lattice value composes with the existing integer-lattice
operations without changing ambient axes.

The exact HNF backend admits the transposed generator matrix and its
intermediate/output height before FLINT expands it. Affine configuration axes
and scalar digits are bounded at the request boundary.
