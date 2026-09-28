# Finite-field elliptic curve base change

`elliptic_curve.finite_field.base_change.compute` transports a nonsingular
short-Weierstrass curve, and optionally one of its points, along an explicit
`FieldEmbedding`. The consumer checks that the curve uses the embedding source
and that the declared image of the source generator is a root of its modulus in
the target field. Thus the map is determined by the supplied presentations and
root, rather than inferred from field names or polynomial moduli.

The transported point retains the transported curve as its exact parent.
Because a field embedding is injective, nonsingularity and the point equation
are preserved; the result is also admitted as a curve and point in the target
presentation.

The source and target must be supported finite fields of equal characteristic,
with source degree dividing target degree. Target presentations are bounded by
the finite-field carrier, and the result size is checked before element mapping.
