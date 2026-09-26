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
