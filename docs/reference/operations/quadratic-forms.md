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

## Integral coefficient content

`quadratic_form.integral_content.compute` accepts polynomial coefficients that
are all integers and returns their nonnegative gcd together with the quotient
form on the same ordered axis. The gcd includes both square and mixed-term
coefficients. An all-zero form has content zero and, by convention, a zero
primitive part. Rational nonintegral coefficients are rejected because an
integral coefficient gcd is not defined for this representation. Input support
is capped at 4096 stored coefficients; the operation is exact and linear in
that bounded support.

## Theta-series prefixes

The native theta-prefix kernel accepts a `RationalQuadraticForm`
whose polynomial coefficients are integers and which is positive definite.
It is not published as a catalog operation: global lattice enumeration
leaves stay outside the public dispatch boundary.
For cutoff `N`, its `coefficients` tuple is exactly
`(r_Q(0), ..., r_Q(N))`, the coefficients of `q^0` through `q^N`; it does not
include an `O(q^(N+1))` term. Variables retain the form's ordered axis, and
vectors are counted with signs and coordinate order, without quotienting by
any symmetry. Nonintegral and non-positive-definite inputs are rejected.
This count-by-representation definition follows the standard theta-series
contract documented by [Sage's quadratic-form reference](https://doc.sagemath.org/html/en/reference/quadratic_forms/sage/quadratic_forms/quadratic_form.html).

Completeness follows from an exact finite box. Let `C` be twice the half-polar
Gram matrix, so `Q(x) = x^T C x / 2`. For every vector with `Q(x) <= N`,
positive definiteness and Cauchy-Schwarz imply
`x_i^2 <= 2*N*(C^-1)_ii`. The kernel computes these rational bounds from exact
principal minors and diagonal cofactors, then enumerates every integer point
in the resulting box. The zero-dimensional form is positive definite by the
empty-matrix convention and has prefix `(1, 0, ..., 0)`.

Admission caps dimension at 7, cutoff at 512, the proved box at 100,000
vectors, total determinant and evaluation work at 2,000,000, determinant
intermediate height at 2,000 decimal digits, and the retained source plus
prefix at 64,000 aggregate decimal digits. Vector and output limits are
checked before lattice enumeration. These bounds are operation limits, not
mathematical limits on theta series.

`quadratic_form.bilinear_pairing.compute` takes two vectors on the same axis
and returns the full polar value

```text
B_Q(x,y) = Q(x+y) - Q(x) - Q(y)
         = sum_i 2*a_i*x_i*y_i
           + sum_{i<j} c_ij*(x_i*y_j + x_j*y_i).
```

This is the full polar pairing, not the half-polar coefficient matrix. The
operation checks both axes and admits coefficient/vector denominator growth
before computing the exact rational result.

`quadratic_form.rational_diagonalization.compute` returns a congruence
diagonalization `D = P^T A P`. `source_axis` names the rows of `P` in the
original form coordinates; `basis_axis` names its columns and the returned
diagonal coefficients. The matrix entries are exact rationals, so the result
retains the coordinate transport needed to reconstruct `D`.

The operation admits its dense work and a common-denominator/Hadamard bound on
all minor ratios before elimination. It rejects dimensions above 64 and exact
coefficient growth beyond the operation's intermediate and output envelopes.
Already diagonal forms use the identity change and retain the form's original
coefficient height.

## Orthogonal direct sums

`quadratic_form.direct_sum.compute` accepts an ordered finite tuple of
`RationalQuadraticForm` values over the same declared domain `QQ`. It combines
their polynomial coefficients into one form on disjoint coordinate axes, so
the resulting value is

```text
Q_0(x_0) + Q_1(x_1) + ... + Q_(r-1)(x_(r-1)).
```

The generated output labels have the form `qf<factor>_<coordinate-index>`.
This avoids collisions when source factors reuse local labels; `source_forms`
retains those original labels and factor order. The result also returns one
coordinate inclusion and one coordinate projection matrix per factor. These
are the standard block-coordinate maps from and to the combined coordinate
space, with map order matching `source_forms`.

The empty family gives the zero-dimensional zero form, the additive identity
for this operation. Aggregate dimension is bounded by 128 and aggregate
polynomial support by 4096 terms before the kernel builds the block result.
Under those envelopes every retained coefficient carries at most 256 digits
per component and the block maps are 0/1, so the maximum legitimate output is
4,341,760 aggregate decimal digits, below the operation's 8,000,000-digit
bound; the result constructor re-establishes that bound for deserialized
values whose dense map entries are caller supplied.
This QQ operation does not coerce integral forms, join equal-named variables,
or claim a lattice isometry; integral-domain promotion needs an explicit
typed map in a later slice.

## Coordinate restrictions

`quadratic_form.coordinate_restriction.compute` accepts a `QQ` form and an
ordered subset of its existing axis labels. The result retains the source
form, returns the restricted form with labels in the caller's requested order,
and gives the exact inclusion matrix from target coordinates into source
coordinates. Repeated or foreign labels are rejected. An empty selection gives
the zero-dimensional zero form and a source-dimension by zero inclusion matrix;
an empty source form is handled the same way.

The restriction is defined by setting every unselected source coordinate to
zero. This is a coordinate-subspace operation; arbitrary subspace restrictions
remain represented by the existing pullback operation with an explicit matrix.
Axis dimension is capped at 128 and polynomial support at 4096 terms; under
those envelopes the source-retaining result's aggregate output digits are
bounded before the inclusion matrix or restricted polynomial is built, and
the result constructor re-establishes the bound for deserialized values.

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

## Finite quadratic Gauss sums

`quadratic_form.finite_gauss_sum.compute` accepts an integral polynomial form
and a modulus `m` and returns

```text
sum_{x in (Z/mZ)^n} zeta_m^Q(x),  zeta_m = exp(2*pi*i/m).
```

The result is the exact element in Jacobian's canonical power basis for
`QQ[zeta_m]`, together with the complete histogram of `Q(x) mod m` and its
total `m^n`. Thus the cyclotomic value is determined by the returned histogram
and the operation's explicit additive-character convention. For `m=1`, the
field convention is `QQ[zeta_1]=QQ` and the sole additive-character value is
one. Nonintegral coefficients are rejected because they do not define a
function on residue classes under this contract.

The kernel first admits the full state count (at most 2,000,000), modulus
(at most 64), polynomial support (at most 4096 terms, which the result
retains and every enumerated state evaluates), their product as kernel work
(at most 2,000,000 term evaluations), a conservative exact
coefficient-growth bound, and the retained source plus canonical output as
aggregate decimal digits (at most 1,000,000), so a state count of one never
admits a response that outgrows the envelope. It then
computes the modular profile once and reduces its histogram polynomial modulo
the canonical cyclotomic polynomial. The exact output is carried by
`RationalCyclotomicElement`; no floating approximation or separate residue
enumeration is used. The finite sum and histogram contract is consistent with
the exact quadratic-form interfaces documented by [Sage](https://doc.sagemath.org/html/en/reference/quadratic_forms/sage/quadratic_forms/quadratic_form.html).
