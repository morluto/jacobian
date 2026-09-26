# Finite-dimensional Lie algebra adjoint representation

`lie_algebra.adjoint_representation.compute` returns the matrices of the
adjoint action for a finite-dimensional Lie algebra over `QQ`. The request uses
the same ordered basis and sparse structure constants as the other
`lie_algebra.*` operations. The result retains the source algebra and contains
one matrix `ad(b_i)` for each basis element `b_i`, in source order.

Matrices act on column coordinate vectors. Column `j` of `ad(b_i)` is the
coordinate vector of `[b_i, b_j]`. The operation accepts dimensions up to the
shared Lie algebra limit of 8. This operation admits every basis-triple
Jacobi identity and matrix work before constructing its result. Jacobi gives the representation law
`[ad(x), ad(y)] = ad([x, y])`; the exact `sl2(QQ)` regression replays this
identity directly.

The operation produces ordinary rational matrices, without a certificate or a
classification result. It does not attempt to find irreducible
representations.
