# Finite Dirichlet-character group enumeration

`dirichlet_character.group.enumerate.compute` takes one canonical finite unit-
group value, rechecks its modulus, generators, orders, and unit-coordinate
table, and returns the complete dual group as one shared parent plus ordered
coordinate tuples. A row `(c_1, ..., c_r)` is the exact dual coordinate of one
character on the group's canonical cyclic generator axes. Rows are in
lexicographic order, including the zero tuple for the principal character.

The family stores the parent once instead of repeating its unit table inside
every character. A caller can construct any member with
`dirichlet_character.compute` by supplying the returned parent and one row.
The operation enumerates characters; it does not materialize their value
tables or claim conductor or parity classifications.

The exact admitted range is the existing character-group range: modulus at
most 2,048 and a complete unit table with at most 2,048 entries. Coordinate
work and the full serialized family are preflighted; the result must fit the
canonical 10 MiB output limit. The carrier validates that all possible dual
coordinates occur exactly once in canonical order. Sage describes a
Dirichlet character as a homomorphism from the finite unit group to roots of
unity and exposes complete group enumeration with `DirichletGroup(N).list()`
([Sage Dirichlet-character reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html)).

[Number-theory operations](index.md) · [Tool surface](../../tools.md)
