# Finite permutation characters

`group.permutation_character.compute` accepts a bounded finite permutation
action and returns its exact permutation character on the generated group's
complete conjugacy-class partition. Each class value is the number of domain
points fixed by a representative. The `FiniteCharacter` value extends the
shared `FiniteClassFunction` carrier, retaining the original action, concrete
permutation group, and complete class partition.

The action domain has at most 50 labelled points and at most 50 generators.
The operation enumerates no more than 64 group elements; a 65th discovered
element causes a resource refusal before class enumeration. Once the group
order is known, the exact conjugation and fixed-point work and the complete
output envelope are admitted before building the class table. The result uses
rational cyclotomic values because every permutation-character value is an
integer.

For the natural action of `S3` on three points, the canonical class values are
`(3, 1, 0)` on the identity, transpositions, and 3-cycles. A trivial group
acting on three points has the one-value character `(3)`.

Decoded `FiniteCharacter` values recheck that the retained partition is the
complete class partition of the action group and that every value matches the
fixed-point count. The same checks apply to direct construction and updated
model copies.
