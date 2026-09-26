# First-order differential-operator least common left multiple

`ore.differential.operator.first_order_lclm.compute` accepts two first-order
differential Ore operators over `QQ(x)` with bounded polynomial coefficients
in `QQ[x]`. It returns a common left multiple `C` and left multipliers `U,V`
such that `C=U*L_left=V*L_right`. The coefficients are written on the left and
operator composition acts from right to left. The result binds both inputs and
the multipliers, so the defining identity can be checked by exact operator
multiplication.

For

\[
A=a_1D+a_0,\qquad B=b_1D+b_0,
\]

write `Delta=a0*b1-a1*b0`. If `Delta=0`, the operators are proportional over
`QQ(x)` and one order-one input is already their least common left multiple.
Otherwise choose `U=b1*D+u0` and `V=a1*D+v0`. Using the Weyl relation
`D*a=a*D+a'`, equality of the order-one and constant coefficients gives

\[
a_1u_0-b_1v_0=a_1(b_1'+b_0)-b_1(a_1'+a_0),
\]
\[
a_0u_0-b_0v_0=a_1b_0'-b_1a_0'.
\]

The nonzero determinant `Delta` determines `u0,v0` exactly, producing an
order-two common left multiple. It is least: an order-one common multiple
would require the two inputs to be proportional. No minimality claim is made
for any broader operator domain.

This is the differential Ore algebra `QQ(x)<D>` introduced by the relation
`D*a=a*D+a'`; see Ore, [“Theory of Non-Commutative Polynomials”](https://doi.org/10.2307/1968173),
*Annals of Mathematics* 34 (1933), 480–508. The formula above is derived
directly and tested against both the commutation identity `xD=Dx-1` and the
two multiplier equations.

Inputs are limited to first order, coefficient degree at most four, at most
four polynomial terms per coefficient, and small exact scalar height. Rational
function coefficients are preserved in the returned multipliers and common
multiple; they are not accepted as input coefficients for this slice.

[Polynomial operations](index.md) · [Operation catalog](../../tools.md)
