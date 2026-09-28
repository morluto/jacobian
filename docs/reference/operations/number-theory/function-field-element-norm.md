# Exact relative norm of a function-field element

`function_field.element.norm.compute` returns the field norm

```text
N_(L/GF(p)(x))(a) in GF(p)(x)
```

for an element `a` of one admitted finite separable extension
`L = GF(p)(x)[y]/(f)`. The result retains the exact presented extension and
the input element; its value is the canonical reduced rational function in
the existing prime-field rational-function carrier. The degree-one rational
function field uses the identity norm.

The norm is the determinant of the base-field-linear map given by
multiplication by `a`. Its matrix uses the power basis `1,y,...,y^(n-1)` and
the defining relation `f(y)=0`; the exact determinant is computed over
`GF(p)(x)`. Admission bounds coefficient degrees, estimated exact arithmetic
work, and serialized result size before rational-function expansion. This
definition applies to finite extensions; see the [Stacks Project, Trace and
norm](https://stacks.math.columbia.edu/tag/0BIE).

The contract covers the current prime-constant-field carrier and its bounded
single extension. It does not construct an extension tower or a map to a
different base field.

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)
