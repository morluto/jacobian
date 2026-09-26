# Exact cellular-sheaf Hodge Laplacians

[`cellular_sheaf.hodge_laplacians.compute`](../../tools.md) computes the exact
degreewise Hodge Laplacian of a finite cellular sheaf over `QQ`. Each stalk
uses its standard coordinate inner product, represented by the identity Gram
matrix on the stalk's declared basis. The degreewise cochain metric is the
orthogonal direct sum of these stalk metrics in the returned cochain axes.

For the cochain coboundary `delta_k : C^k -> C^(k+1)`, the operation returns
the up and down Laplacians and their Hodge sum:

```text
L_k^up   = delta_(k-1) delta_(k-1)^T
L_k^down = delta_k^T delta_k
L_k      = L_k^up + L_k^down
```

with the missing end-degree term taken as zero. All entries are rational and
all matrices and harmonic basis vectors are bound to the sheaf's degreewise
cochain bases. The harmonic kernel dimension is checked against the exact
cellular-sheaf cohomology dimension.

The identity Gram forms are positive definite over the real interpretation of
`QQ`, so these exact kernels represent cohomology. `GF(p)` sheaves are not
supported by this operation. Nonstandard stalk metrics and their Gram
transport are not yet represented.

Before assembling cohomology or Laplacian matrices, the operation admits at
most 512 total cochain coordinates, 65,536 Laplacian matrix entries, 4,000,000
matrix contraction terms, and a conservative 8,000,000-character result bound.
An over-bound request is rejected before exact matrix expansion.
