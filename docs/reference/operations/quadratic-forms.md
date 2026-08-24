# Exact rational quadratic forms

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
