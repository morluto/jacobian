# Exact inversion of function-field elements

`function_field.element.inverse.compute` returns the multiplicative inverse of
a nonzero element of the presented finite function field
`GF(p)(x)[y]/(f(y))`. The returned `FiniteFunctionFieldElement` retains the
identical field parent and uses the existing reduced power-basis coordinates.

The kernel applies extended Euclid to the element polynomial and the monic
defining polynomial over `GF(p)(x)`. A conservative resultant-style degree
bound, operation-work bound, and exact serialized-result size bound are checked
before that Euclidean expansion. Inputs whose admitted coefficient envelope
cannot cover the conservative bound receive a typed resource rejection. Zero
receives a domain error because it is not invertible. The field itself must
pass the same characteristic, separability, irreducibility, and extension
checks as the other arithmetic operations.

## Example

In `GF(2)(x)[y]/(y^2+y+x)`, the inverse of `y` is `(y+1)/x`, since
`y(y+1)=x`. Multiplying the returned inverse by `y` gives the unit with the
same field parent.

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)
