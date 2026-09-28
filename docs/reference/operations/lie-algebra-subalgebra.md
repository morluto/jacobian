# Induced Lie algebra on a subalgebra

`lie_algebra.subalgebra.compute` takes a finite-dimensional Lie algebra over
`QQ`, a canonical RREF subspace on its ordered basis, and labels for the
candidate rows. It checks that the subspace is bracket-closed, then returns
the induced structure-constant algebra and the inclusion matrix whose rows
are those basis vectors in source coordinates.

For candidate rows `u_i`, the result constants are the unique coefficients
`c_ij^k` satisfying `[u_i,u_j] = sum_k c_ij^k u_k`. The operation computes the
source bracket exactly, verifies the bracket lies in the candidate span, and
uses RREF pivot coordinates to recover those coefficients. The output basis
follows candidate row order; labels name these induced basis vectors.

The operation establishes the source and induced Jacobi identities and bounds
dimension, rational heights, closure work, and output. A subspace that is not
closed returns a domain error. A zero-row candidate is the zero subalgebra:
the empty induced basis labels are accepted exactly when the candidate has no
rows, and the result retains the zero-dimensional algebra
`FiniteDimensionalLieAlgebra(basis=(), structure_constants=())`. That value is
a canonical carrier and composes unchanged with every consuming Lie-algebra
operation, which admit the empty basis axis.
