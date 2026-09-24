# Exact inverse of a Dirichlet character

`dirichlet_character.inverse.compute` returns the inverse of one Dirichlet
character in the same finite dual-group parent. It negates the exact dual
coordinates modulo their generator orders. Pointwise multiplication of the
input and result is the principal character on the same group.

Dirichlet-character values are roots of unity on units and zero off the unit
group, so the inverse character also equals the complex-conjugate character.
The inverse operation names the finite-group postcondition directly and
composes with `dirichlet_character.multiply.compute`; it does not return a
separate proof record.

The operation validates the complete source group and character through the
same owner admission used by the native API. Its work is linear in the number
of dual coordinates, and its result retains the complete group parent and
canonical coordinate tuple.

[Number-theory operations](index.md) · [Exact conjugation of Dirichlet characters](dirichlet-character-conjugate.md) · [Exact Dirichlet-character powers](dirichlet-character-power.md)
