# Finite-field elliptic Frobenius data

`elliptic_curve.finite_field.frobenius.compute` returns the exact cardinality
`N`, Frobenius trace `t = q + 1 - N`, determinant `q`, characteristic
polynomial `X² - tX + q`, and discriminant `t² - 4q` for a nonsingular
short-Weierstrass curve over the admitted finite field. It also classifies the
curve as ordinary or supersingular.

The input domain is the existing exact short-Weierstrass model in
characteristic `p > 3`; the exact quadratic-character count is bounded by the
finite-field work envelope. Supersingularity uses the finite-field criterion
that an elliptic curve over a field of characteristic `p` is supersingular
exactly when `p` divides the Frobenius trace. See [MIT 18.783, Lecture 14,
§14.1](https://math.mit.edu/classes/18.783/2015/LectureNotes14.pdf).

The result retains the complete source curve and checks consistency among its
field order, cardinality, trace, polynomial, discriminant, and classification.
This operation contributes the Frobenius-classification slice to issue #1781;
it does not provide certified non-exhaustive point counting or Schoof/SEA
trace certificates.

[Number theory operations](index.md) · [Tool surface](../../tools.md)
