# Root lattice inside the weight lattice

`root_system.root_weight_lattice_embedding.compute` returns the simple-root
lattice `Q` as an exact `IntegerLattice` sublattice of the fundamental-weight
lattice `P`. The result retains the canonical finite Cartan datum, both
lattices, and the inclusion matrix, so its values can be passed directly to
`lattice.sublattice_index.compute`.

The shared ambient integer coordinates are the ordered fundamental-weight
basis. With the convention `A[i,j] = <alpha_i^vee, alpha_j>`, column `j` of
the Cartan matrix gives simple root `alpha_j` in fundamental-weight
coordinates. `IntegerLattice` stores basis vectors as rows, so the retained
embedding is `A^T`; the parent weight lattice has identity basis. Thus the
operation makes the relation `Q = A^T P` explicit without discarding the
Cartan parent or its ordered axes.

Rank is at most 8, matching the finite Cartan datum. The presentation carries three additional rank-square matrices beyond the three
matrices retained by the Cartan datum, plus the Cartan datum axes. Work is cubic
in rank and the serialized output has a quadratic cell bound, both capped by the
rank envelope. The operation rejects noncanonical or
non-finite Cartan data before materializing the lattice bases.

For type A2 the index of `Q` in `P` is 3; composing the returned values with
the generic exact sublattice-index operation yields the quotient `P/Q`.

[Root-system operations](root-coroots.md) · [Tool surface](../../tools.md)
