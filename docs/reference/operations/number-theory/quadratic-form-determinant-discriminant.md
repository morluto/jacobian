# Rational quadratic-form determinant and signed discriminant

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`quadratic_form.determinant_discriminant.compute` returns the determinant of the full polar Gram matrix and its dimension-signed discriminant. For `Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j`, the polar matrix has `G_ii=2a_i` and `G_ij=c_ij`; unlike the coefficient matrix used for `Q=x^TAx`, off-diagonal entries are not halved. The signed scalar is `(-1)^(n(n-1)/2)*det(G)`.

The kernel uses fraction-free Bareiss elimination over an integer matrix obtained by clearing each row's denominators. Dimension, stored-matrix entries, support, work, exact intermediate height, output height, and retained source digits are admitted before elimination. For `x^2+xy+y^2`, the polar determinant is `3` and the signed discriminant is `-3`, matching the classical binary discriminant `b^2-4ac`.
