# Integral quadratic-form content

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`quadratic_form.integral_content.compute` returns the nonnegative gcd of the coefficients of an integral polynomial quadratic form and the associated primitive quotient. It divides both diagonal and cross coefficients by that gcd and retains the source axis.

The zero polynomial has content zero and remains zero. Only integer polynomial coefficients are accepted; retained support is bounded by 4,096 coefficients. For `6x^2+9xy`, the content is `3` and the primitive quotient is `2x^2+3xy`.
