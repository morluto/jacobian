# Composita of quadratic splitting fields

`number_field.galois.compositum.compute` computes the compositum of two
source-bound splitting fields over `QQ`, each of degree at most two. The result
is one exact simple-field presentation together with an embedding of each
input field. The maps retain the concrete inclusions, so an isomorphic field
presentation alone is never treated as the compositum result.

If an input is `QQ`, its inclusion is the unique rational-field map. If both
inputs are the same quadratic field up to a rational square change of radical,
the result reuses one quadratic presentation and maps the other generator into
it. For distinct quadratic classes, write the inputs as `QQ(sqrt(D1))` and
`QQ(sqrt(D2))`, where each `Di` is the discriminant of its retained quadratic
generator. The operation uses
`theta = sqrt(D1) + sqrt(D2)` with minimal polynomial
`X^4 - 2(D1 + D2) X^2 + (D1 - D2)^2` and returns exact inclusion maps expressed
in the power basis of `theta`. Independence of the two quadratic classes gives
degree four, and the two maps generate the returned field.

This is a bounded compositum operation, not a general normal-closure or
intermediate-field solver. The current splitting-field inputs have degree at
most two, so the output degree is at most four. The field presentation and
embeddings use Jacobian's exact simple-number-field carriers. The scope follows
the standard field-theoretic compositum and primitive-element treatment; see
[Howie, *Fields and Galois Theory*](https://link.springer.com/book/10.1007/978-1-84628-181-5).
