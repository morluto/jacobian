# Plane-curve strict-transform divisor classes

`algebraic_geometry.plane_curve.strict_transform_class.compute` maps a
nonzero homogeneous polynomial `F` of degree `d` and a labelled blow-up
surface of the projective plane to the existing exact `BlowupDivisorClass`
value

```text
d H - sum_i m_i E_i
```

Here `m_i` is the local multiplicity of `F = 0` at the corresponding
projective point. The operation finds it as the least total Taylor degree
with a nonzero coefficient after setting a nonzero projective coordinate to
one. The request gives an explicit permutation from point coordinate order
`[X0:X1:X2]` to the polynomial variable axis.

The returned class composes directly with
`algebraic_geometry.blowup_p2.intersection.compute`, which evaluates
`d e - sum_i m_i n_i` on the same labelled parent. The operation does not
find singular points or choose blow-up centers; callers provide the surface.
It supports at most 16 distinct proper rational points, degree 12, 64 source
terms, and bounded rational coefficient and point heights. The full point
family, polynomial coefficients, Taylor-coefficient growth, work, and class
output are admitted before surface canonicalization or multiplicity
calculation.

The class formula is the standard strict-transform relation for blowing up
points on a plane curve: a degree-`d` curve with multiplicities `m_i` has
class `dH - sum_i m_i E_i` on the blow-up ([reference](https://www.mdpi.com/2227-7390/12/24/3952)).
For the scheme-theoretic definition of strict transform under a blow-up, see
[Stacks Project, Section 71.18](https://stacks.math.columbia.edu/tag/0861).
