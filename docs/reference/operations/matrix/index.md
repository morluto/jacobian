# Matrix operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Matrix operations use bounded typed integer, rational, and symbolic
rational-function matrices and return their values inline. The live catalog is
authoritative for request and result schemas. It includes determinant, rank,
RREF, nullspace, products, inverses, trace, adjugate, characteristic
polynomial, and Smith normal forms.

The direct normal-form operations are `matrix.normal_form.smith.compute` and
`matrix.normal_form.smith.certified.compute`. The certified Smith result keeps
the matrices needed to inspect (D = UAV) in the returned value; it does not
publish a record or require later retrieval.

- [Exact rational matrix determinants](matrix-rational-determinant.md)
- [Exact subsystem-aware Hermitian matrices](subsystem-hermitian-matrices.md)
- [Exact symbolic matrix products](matrix-symbolic-products.md)
