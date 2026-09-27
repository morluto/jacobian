# Applying an exact number-field embedding

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`number_field.embedding.apply_exact.compute` transports one element under a
specified embedding between two simple extensions of `QQ`. Each field uses
the shared primitive presentation

```text
QQ(alpha) = QQ[x] / (f).
```

The request specifies the target-field image of the source generator. The
operation recognizes both defining polynomials as irreducible, evaluates the
source polynomial exactly at the proposed image in the target quotient, and
rejects the request unless the result is zero. Irreducibility makes this
unital map injective. It then evaluates the source element in the same target
quotient and returns both that exact element and a reusable
`SimpleNumberFieldEmbedding` value.

This is a supplied-map operation: it does not search for roots or choose an
embedding. Its scope is absolute fields over `QQ`, degrees at most six, and
polynomial and coordinate components at most 32 decimal digits. A conservative
coordinate-growth estimate is checked before field arithmetic; the bound admits
the identity embedding up to degree six, so degree seven and eight presentations
are rejected by the degree admission rather than by an unusable estimate. The result
retains the exact source and target parents, so the target coordinates can be
composed with the existing `SimpleNumberFieldElement` and relative
trace/norm operations.

For example, if `beta^4 = 2`, the image `beta^2` satisfies the defining
polynomial of `QQ(sqrt(2))`; therefore the operation maps `sqrt(2)` to
`beta^2` and `1 + sqrt(2)` to `1 + beta^2` exactly.

This provides a small reusable map contract for field/action work. It does not
construct splitting fields, normal closures, automorphism groups, or fixed
fields. Sage's exact number-field morphisms also specify a map by generator
image and require that image to satisfy the source defining polynomial; see
[Sage number-field morphisms](https://doc.sagemath.org/html/en/reference/number_fields/sage/rings/number_field/number_field_morphisms.html).
