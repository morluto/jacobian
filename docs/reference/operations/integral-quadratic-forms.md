# Integral quadratic forms

`IntegralQuadraticForm` is the canonical polynomial carrier over `ZZ`:

```text
Q(x) = Σᵢ aᵢ xᵢ² + Σᵢ<ⱼ cᵢⱼ xᵢ xⱼ
```

The stored diagonal and cross coefficients are the polynomial coefficients.
The ordered coordinate axis is part of the value; cross terms use zero-based
indices `left < right`. Thus an odd cross coefficient remains an odd
polynomial coefficient even though the associated `Q(x)=xᵀAx` matrix has
half-integral off-diagonal entries.

`quadratic_form.integral.rational_extension.compute` applies the explicit
coefficient map `ZZ → QQ`. Its `IntegralQuadraticFormInclusion` retains both
source and target forms, the shared coordinate axis, and the map identity. The
target is the coefficient-wise image and can be passed directly to rational
quadratic-form operations after JSON serialization. There is no implicit
coercion from `QQ` back to `ZZ`.

The carrier admits at most 128 coordinates, 2,048 nonzero polynomial terms,
and 256 decimal digits per coefficient. The inclusion operation bounds the
combined serialized source and target size before constructing the `QQ` value.
An empty axis is the zero-dimensional form; a degenerate or zero form on a
nonempty axis remains a valid integral form.

`quadratic_form.parity_profile.compute` maps the polynomial coefficients into
`F₂` and preserves the ordered axis in a `ModularQuadraticPolynomial`. Its
diagonal residues are the basis norms `Q(eᵢ) mod 2`. For `i < j`, its mixed
residue is the polar pairing
`B(eᵢ,eⱼ) = Q(eᵢ+eⱼ) - Q(eᵢ) - Q(eⱼ) mod 2`, equal to the stored coefficient
of `xᵢxⱼ` under this polynomial convention. Zero mixed residues are omitted
canonically. `all_basis_norms_even` reports whether every basis-vector norm is
even; it does not decide whether all lattice vectors have even norm.

The profile reuses coefficient-reduction admission: work depends on source
coefficient digits and support, and the output is bounded by the same axis and
term limits, with residues in `{0,1}`. Its allocation check runs before the
modular polynomial is constructed. The zero-dimensional profile has no
diagonal residues and `all_basis_norms_even = true` by vacuity.
