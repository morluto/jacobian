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

The canonical source value establishes Jacobi at construction. The operation
bounds dimension, rational heights, closure work, and output; construction of
the induced algebra establishes its Jacobi identity before returning it. A
subspace that is not closed returns a domain error. The current Lie-algebra
carrier requires a nonempty basis, so the zero subalgebra has no induced
algebra representation.
