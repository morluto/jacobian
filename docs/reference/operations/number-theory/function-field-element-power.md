# Exact powers of function-field elements

`function_field.element.power.compute` returns the exact nonnegative integral
power of one element in its presented finite function field
`GF(p)(x)[y]/(f(y))`. The result is a canonical `FiniteFunctionFieldElement`
with the same field parent. Exponent zero returns the parent unit, including
when the element is zero; exponent one returns the canonical input value.

The kernel uses binary powering with exact arithmetic in `GF(p)(x)` and exact
reduction modulo the defining polynomial. Before coefficient arithmetic, it
admits every multiplication in the binary schedule against a conservative
coefficient-degree bound, the aggregate work bound, and the serialized result
size. Inputs whose intermediate bound exceeds the supported coefficient
envelope receive a typed resource rejection. Exponents must be nonnegative;
negative powers compose with `function_field.element.inverse.compute`.

In `GF(2)(x)[y]/(y^2+y+x)`, the defining relation gives `y^2=y+x`, and
therefore `y^4=y+x+x^2`. The operation returns those exact reduced coordinates.

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)
