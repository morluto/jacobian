# Disjoint union of finite relational structures

`relational_structure.disjoint_union.compute` takes two canonical
`FiniteRelationalStructure` values over the same ordered ranked signature. It
returns their disjoint union as another canonical structure and includes both
component inclusions.

Left carrier label `i` stays `i`; right label `j` becomes `|A| + j`. For a
positive-arity relation `R`, the result contains exactly the left tuples and
the offset right tuples. Thus no tuple mixes elements from the two components.
For a nullary relation, the result is true when it is true in either component
and false when it is false in both. The combined carrier must fit the
canonical 64-label bound, and each output relation table must fit its 4,096-row
bound. These cardinalities and the row/coordinate reconstruction work are
admitted before output construction; the derived work ceiling is 720,960
visits.

The inclusions are homomorphisms. They reflect positive-arity relations on
each component; a false nullary relation in one component can become true in
the union when it is true in the other, so that inclusion does not reflect
that nullary relation. For every target structure `C` over the same signature,
restriction along the inclusions gives the finite universal-property bijection

```text
Hom(A ⊔ B, C)  ≅  Hom(A, C) × Hom(B, C).
```

The operation returns ordinary reusable structure and map values; it does not
search for or enumerate any of these homomorphisms.
