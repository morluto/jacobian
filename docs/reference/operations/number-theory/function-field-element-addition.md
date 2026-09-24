# Exact addition of function-field elements

`function_field.element.add.compute` adds two elements in the same presented
finite function field

```text
GF(p)(x)[y]/(f(y)).
```

An element is represented by its reduced power-basis coordinates in
`GF(p)(x)`. The operation adds the corresponding rational functions exactly,
then returns the direct `FiniteFunctionFieldElement` value with that same field
parent. It does not attach a trace or proof ledger; each rational-function
coordinate is reduced to coprime numerator and monic denominator by the
canonical `GF(p)(x)` arithmetic.

Both input elements must carry the identical field presentation. The consumer
re-admits the field's characteristic, separability, irreducibility, and
extension bounds, and canonicalizes the input coordinates before estimating
the result. Admission bounds the rational-function support, coefficient
degree, Euclidean normalization work, and serialized element size before
performing coordinate addition. If the conservative cross-product bound can
exceed the degree-12 coefficient envelope, the operation returns a typed
resource rejection before expanding that sum.

This operation supports the same prime-constant-field and bounded single
extension carrier as the existing element multiplication operation. It does
not transport elements between different presentations or construct field
maps.

## Example

In characteristic two, `y + (y + x) = x` in
`GF(2)(x)[y]/(y^2 + y + x)`. The result is the one-coordinate rational
function-field element for `x`, bound to the original field.

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)
