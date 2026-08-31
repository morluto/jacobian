# Exact quadratic forms

[Documentation home](../../index.md) · [Tool surface](../tools.md)

`quadratic_form.evaluate.compute` accepts one canonical rational quadratic form
and one rational coordinate vector on exactly the same ordered axis. It returns
the source-bound exact rational value.

The form uses polynomial coefficients, not an implicit symmetric-matrix
convention:

```text
Q(x) = sum_i a_i x_i^2 + sum_{i<j} c_ij x_i x_j.
```

`diagonal_coefficients[i]` stores `a_i`; zero cross coefficients are omitted,
and nonzero cross terms are ordered by their coordinate-index pair. The
associated polar matrix is derived as `B_ii = 2 a_i`, `B_ij = c_ij`, so its
half-polar Gram matrix has off-diagonal entries `c_ij / 2`. Derived matrices
are deliberately not independently accepted or returned by this leaf.

Evaluation is direct exact rational arithmetic. It supports degenerate and
indefinite forms because one value at one supplied vector is always finite;
representation numbers and theta prefixes are not part of this contract.

## Proper classes of positive-definite integral binary forms

`number_theory.binary_quadratic_form.reduced_classes.compute` returns each
proper class as a `ProperBinaryQuadraticFormClass`. Its representative is the
unique canonical Gauss-reduced primitive form

```text
Q(x,y) = a*x^2 + b*x*y + c*y^2,
|b| <= a <= c,
b >= 0 when |b| = a or a = c.
```

The class retains its quadratic-order discriminant through the representative;
nonfundamental discriminants refer to proper classes of the corresponding
quadratic order, not automatically to the maximal order of its fraction field.

`number_theory.binary_quadratic_form.class_compose.compute` multiplies two
proper classes of the same discriminant. It returns a direct composed form, a
bilinear map in monomial order
`(x1*x2, x1*y2, y1*x2, y1*y2)` satisfying

```text
H(X,Y) = F(x1,y1) * G(x2,y2),
```

and the canonical reduced product with an exact `SL_2(ZZ)` reduction matrix.
Composition is admitted by its own work bounds (direct O(1) Buell
formula and O(log|D|) Gauss reduction), independent of the reduced-class
enumeration budget. The reduced product coefficients are bounded by
`|D|/3`, so the discriminant is admitted only when `|D|/3 <= 10^6`.

The private kernel uses the classical direct Gauss-composition formula with a
ternary Bézout relation, then applies the existing exact Gauss reduction. The
kernel is Jacobian-owned because Python-FLINT does not expose binary-form
composition and PARI/cypari2 is not part of the supported runtime. Backend
objects and algorithm-specific class labels do not cross the public boundary.
