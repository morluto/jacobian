# Rational base inclusion into a function-field extension

`function_field.base_embedding.construct` returns the canonical embedding
`GF(p)(x) -> GF(p)(x)[y]/(f)` for one admitted finite separable extension. The
source uses the target's characteristic and rational-variable name; because
the constant field is the prime field `GF(p)`, its unital inclusion is unique.
The carrier also stores the exact target-field image of `x` as a power-basis
element and checks that it is `(x, 0, ..., 0)`. The typed result retains both
exact field parents so later operations can bind elements and divisors to the
intended source or target.

`function_field.base_embedding.apply` consumes this embedding with a source
element and returns the same rational function as the constant power-basis
coordinate of a target element. The result retains the embedding and both
source/image parents, so the image composes directly with target-field
arithmetic such as `function_field.element.multiply.compute`. Applying a
deserialized embedding rechecks its canonical variable image and re-admits the
target extension before returning the transported value.

This is only the natural base-field inclusion. It does not construct arbitrary
maps, identify isomorphic presentations, describe towers, or map places and
divisors. The target extension is checked for primality of the characteristic,
separability, irreducibility, and the function-field resource envelope before
the map carrier is returned.

Sage's function-field morphism reference documents general generator-image
maps, which is broader than this one canonical inclusion:
[Sage function-field morphisms](https://doc.sagemath.org/html/en/reference/function_fields/sage/rings/function_field/maps.html).

## Example

For `GF(2)(x)[y]/(y^2+y+x)`, the operation returns the source rational field
`GF(2)(x)` and the exact extension field as the target. Both carry the same
named `x`; the inclusion is fixed by that data.

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)
