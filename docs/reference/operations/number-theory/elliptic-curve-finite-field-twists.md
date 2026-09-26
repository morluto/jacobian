# Elliptic curve quadratic twists over finite fields

The public
`elliptic_curve.finite_field.quadratic_twist_relation.compute` operation takes a
nonsingular short-Weierstrass curve
\(E:y^2=x^3+Ax+B\) over a finite field of characteristic greater than three.
It chooses the first nonsquare `d` in the field's canonical base-p
power-basis encoding and returns a source-bound relation containing the model
\[
E^{(d)}:y^2=x^3+d^2Ax+d^3B.
\]

This is the unique nontrivial quadratic twist up to isomorphism over the
source field. The chosen `d` makes the returned coefficient pair canonical for
the declared field presentation. The result retains that field; pass its
`twisted_curve` directly to point enumeration, cardinality, and extension-count
operations.
The expected independent cardinality relation is
\[
#E(\mathbb F_q)+#E^{(d)}(\mathbb F_q)=2(q+1),
\]
equivalently the two Frobenius traces are negatives. PARI's reference documents
the unique nontrivial twist for curves over a finite field in its
[`elltwist` operation](https://pari.math.u-bordeaux.fr/dochtml/ref/Elliptic_curves.html#elltwist).

The implementation admits fields of order at most 4,096. It bounds the
worst-case scan and exponentiation work, and the maximum serialized curve
shape, before searching for the nonsquare. Characteristics two and three,
singular cubics, and larger fields are outside this operation's contract.

See the [source-bound twist relation](elliptic-curve-quadratic-twist-relation.md) for the complete result contract.
