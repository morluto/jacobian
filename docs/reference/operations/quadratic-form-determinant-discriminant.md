# Quadratic form determinant and signed discriminant

[Quadratic forms](./quadratic-forms.md) · [Tool surface](../tools.md)

`quadratic_form.determinant_discriminant.compute` accepts a rational quadratic
form in its canonical polynomial-coefficient convention

```text
Q(x) = sum_i a_i*x_i^2 + sum_(i<j) c_ij*x_i*x_j.
```

The operation returns the determinant of the **full polar Gram matrix**
`G`, defined by

```text
B_Q(x,y) = Q(x+y)-Q(x)-Q(y) = x^T G y
G_ii = 2*a_i
G_ij = c_ij.
```

This matrix is twice the coefficient matrix `A` from `Q(x)=x^T A x`:
`A_ii=a_i` and `A_ij=c_ij/2`. Odd polynomial cross coefficients therefore
make `A` half-integral while `G` retains the original cross coefficient.

The result contains both `det(G)` and the named signed normalization

```text
(-1)^(n*(n-1)/2) * det(G),   n = len(form.axis).
```

In dimension two this is the classical binary polynomial discriminant
`b^2-4ac` for `Q=ax^2+bxy+cy^2`. The returned signed rational is a
representative, not a basis-invariant scalar: under an invertible rational
change `P`, it is multiplied by `det(P)^2`. Its square class is invariant,
and is defined only for a nondegenerate form. Singular forms return zero for
both determinant values. The empty, zero-dimensional form uses `det(0x0)=1`,
so its signed discriminant is also one.

The Gram determinant's square-class invariance follows from the congruence
law `G' = P^T G P`; see the [quadratic-forms reference chapter](https://link.springer.com/chapter/10.1007/978-3-030-56694-4_4).
The [Encyclopedia of Mathematics entry on binary quadratic forms](https://encyclopediaofmath.org/wiki/Binary_quadratic_form)
records the competing binary determinant and polynomial-discriminant
normalizations that motivate returning both named quantities.

The exact fraction-free Bareiss kernel clears each row's rational denominators
and applies Hadamard bounds to all minors and intermediate products before
constructing the integer matrix. The operation admits dimension at most 64,
at most 4096 matrix entries and stored source terms, a bounded cubic work
estimate, and exact output/intermediate heights. These limits are operation
envelopes, not restrictions on mathematical dimension.
