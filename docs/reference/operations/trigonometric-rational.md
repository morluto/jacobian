# Exact trigonometric-rational normalization

[Documentation home](../../index.md) · [Operation references](index.md)

`algebra.trigonometric_rational.normalize` maps a bounded typed expression in
integer-affine sines and cosines to a reduced Laurent rational function over
the distinguished Gaussian-rational field `QQ(i)`. The declared variable axis
is preserved in every returned polynomial.

For `z_j = exp(i x_j)` and an integer exponent vector `m`, the operation uses
the exact identities

```text
cos(m.x + k*pi/2) = (i^k z^m + i^-k z^-m) / 2
sin(m.x + k*pi/2) = (i^k z^m - i^-k z^-m) / (2i).
```

The input AST supports rational literals, addition, multiplication, division,
nonnegative integer powers, and `SIN`/`COSINE` leaves. It is a named,
non-evaluating grammar: arbitrary expression strings, `eval`, and numerical
angle evaluation are outside the contract.

The numerator and denominator are reduced by exact common-factor cancellation
and normalized to one canonical Laurent presentation. `denominator_nonzero`
retains the denominator locus of the source expression before cancellation, so
an identity such as `sin(x)/sin(x) = 1` still records that the source was only
defined where `sin(x)` is nonzero. On the algebraic torus, monomial shifts do
not change a nonzero locus.

The operation admits at most eight variables, 128 AST nodes, 4,096 Laurent
terms, and absolute Laurent exponents of 4,096. Support convolution,
intermediate Gaussian-rational coefficient growth, and canonical output
support are rejected before they exceed the exact envelope. The returned
value is ordinary typed mathematical data and composes directly with later
Laurent-polynomial operations; exact maximization on a real circle is a
separate real-algebra operation.
