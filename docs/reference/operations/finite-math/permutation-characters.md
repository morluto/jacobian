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

Deserialization performs structural checks on the retained action, class
rows, and values; it does not re-enumerate the generated group or replay the
fixed-point computation. Completeness is established by this operation's
producer. A consumer relying on a caller-authored character or class-partition
claim must check the relevant defining relation within its admitted work.
