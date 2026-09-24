# Reversal anti-automorphism of a free algebra

`free_algebra.polynomial.reverse_antiautomorphism.compute` acts on a sparse
polynomial in the declared ordered alphabet of `QQ<X>`. It fixes each rational
coefficient and reverses each monomial word:

```text
rev(sum_w a_w w) = sum_w a_w rev(w)
rev(x_1 ... x_n) = x_n ... x_1
```

This is an anti-automorphism, so it reverses multiplication order,
`rev(fg) = rev(g)rev(f)`, and applying it twice returns the original
polynomial. The operation returns the ordinary canonical
`FreeAlgebraPolynomial`: the rational coefficient ring and ordered generator
alphabet are preserved, and terms are sorted in the polynomial's canonical
degree-lexicographic order after reversal.

Admission bounds support traversal and the conservative serialized result
estimate before reversed terms are constructed. The polynomial value limits
the alphabet to 26 generators, the support to 4096 terms, and each stored word
to 64 letters; this operation has a 4,000,000-unit reversal/sorting-work bound
and a 2 MB output bound.

The operation only reverses words. It does not permute generator labels or
conjugate rational coefficients.
