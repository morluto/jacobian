# Finite-field elliptic point transport

`elliptic_curve.finite_field.isomorphism.transport_point.compute` applies one
caller-supplied short-Weierstrass isomorphism to a point over a single exact
finite-field presentation. For the scaling convention used by the
isomorphism result, the map is

```text
(x, y) -> (u^2 x, u^3 y)
```

and the endpoint coefficients must satisfy `A_target = u^4 A_source` and
`B_target = u^6 B_source`. The operation rechecks the exact field identity,
both nonsingular curve models, the nonzero scaling, the coefficient
identities, and the source point's curve equation before using the map.
Infinity maps to infinity. The result retains the supplied isomorphism and
both source and target points, so it can be serialized without losing either
parent.

The short-model coordinate formula is valid in characteristic greater than
three; the elliptic-curve owner rejects other characteristics. This operation
does not search for an isomorphism. The complete bounded search is provided by
`elliptic_curve.finite_field.isomorphism.decide`.

For short Weierstrass models over a field of characteristic at least five,
isomorphisms fixing infinity have the scaling form above; see [Galbraith,
Harrison, and Moreno, *Constructing Isogenies between Elliptic Curves over
Finite Fields*, Appendix A.2](https://www.cambridge.org/core/services/aop/cambridge-core/content/view/30811E8101E76F7C2A22873EE7080237/S1461157000000097a.pdf/div-class-title-constructing-isogenies-between-elliptic-curves-over-finite-fields-div.pdf).
