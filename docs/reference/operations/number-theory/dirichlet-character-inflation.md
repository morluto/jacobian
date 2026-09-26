# Inflation of a Dirichlet character

`dirichlet_character.inflate.compute` induces a character from source modulus
`d` to a target modulus `m` when `d` divides `m`. On target units it composes
the source character with the reduction map
`(Z/mZ)^* -> (Z/dZ)^*`. The resulting Dirichlet character is zero on target
nonunits, following the standard extension-by-zero convention.

The result retains the source character, the exact target character in the
target group's canonical dual coordinates, and aligned residue arrays giving
the reduction of every target unit modulo `d`. This preserves both character
parents and makes the map used by the induction explicit. The operation does
not claim a map on nonunits: their target character values are zero.

For example, inflating the nonprincipal character modulo `3` to modulus `15`
maps each unit modulo `15` to its residue modulo `3`. The target units that
reduce to `1` have value `1`; those reducing to `2` have value `-1`. Residues
not coprime to `15` have value zero, even when their reduction modulo `3` is a
unit.

Both moduli use the existing `2,048` group-modulus bound. Complete source and
target unit groups and the source-linked output are admitted before the target
character and reduction map are returned. Nonmultiples are rejected. The
definition agrees with Sage's convention that a Dirichlet character is a
homomorphism on units extended by zero on nonunits
([Sage reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html)).
