# Root, coroot, weight, and coweight vectors

The four vector operations return distinct immutable values, each bound to a
canonical finite Cartan datum:

- `root_system.root_lattice.vector.compute` uses the ordered simple-root basis.
- `root_system.coroot_lattice.vector.compute` uses the ordered simple-coroot basis.
- `root_system.weight_lattice.vector.compute` uses the ordered fundamental-weight basis.
- `root_system.coweight_lattice.vector.compute` uses the ordered fundamental-coweight basis.

The datum convention is `A[i,j] = <alpha_i^vee, alpha_j>`. With column
coordinates, the inclusions from root to weight and coroot to coweight
coordinates are respectively `A c` and `A^T d`. The operations
`root_system.root_to_weight_lattice.compute` and
`root_system.coroot_to_coweight_lattice.compute` apply those exact integral
maps and return values in the target lattice. Each map re-admits the Cartan
matrix and checks the serialized symmetrizer and basis maps against the
canonical datum before doing arithmetic.

These maps are inclusions, not identifications: their images can be proper
sub-lattices. No implicit inverse or change of parent is supplied. Reverse
membership and quotient invariant factors are separate operations. This
representation describes the four lattices attached to a finite Cartan datum;
it does not choose an arbitrary isogeny or an intermediate character lattice.

`root_system.connection_index.compute` supplies the root-system-specific
quotient `P/Q`: it returns the nontrivial Smith invariant factors and order of
the weight lattice modulo the root lattice. In the convention above, column
`j` of `A` is the simple root `alpha_j` in fundamental-weight coordinates, so
the quotient is the cokernel of the root-to-weight inclusion matrix `A`.
The operation reuses the exact bounded matrix Smith-normal-form capability and
does not enumerate quotient representatives. Rank is limited to 8, the
admitted Cartan entries are finite type, and the matrix Smith operation applies
its own exact input/work/output admission.

For example, `P/Q` is cyclic of order `n + 1` for `A_n`, cyclic of order 2 for
`B_n` and `C_n`, and for `D_n` is cyclic of order 4 when `n` is odd or a direct
product of two cyclic groups of order 2 when `n` is even. Reducible Cartan data
are handled by applying Smith form to their full block-diagonal matrix, which
canonicalizes the product quotient's invariant factors. The Magma
[Root Data handbook](https://docs.magma-maths.org/LieTheory/RootData/introduction.html)
describes the fundamental group through the Cartan matrix quotient.

The request rank is at most 8. Input coordinates are bounded to 128 bits and
mapped coordinates to 133 bits; work and retained output are admitted before
the matrix-vector product. The basis convention is part of the operation
contract above and the returned datum retains the exact basis maps.

[Operation references](../index.md) · [Tool reference](../../tools.md)
