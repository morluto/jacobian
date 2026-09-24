# Exact conjugation of a Dirichlet character

`dirichlet_character.conjugate.compute` returns the complex conjugate of one
canonical exact character, preserving its exact modulus and unit-group parent.
For a unit `a`, the output satisfies `conj(chi)(a) = conjugate(chi(a))`; for a
nonunit both values are zero. Since all unit values are roots of unity, the
native implementation negates the character's dual-group coordinates modulo
their generator orders.

The exact value table can be checked by composing with
`dirichlet_character.table.compute`. Conjugation is involutive, and the product
of a character with its conjugate is the principal character on the same unit
group. The operation does not enumerate character families or change parents.

Sage documents the same exact complex-conjugate operation on Dirichlet
characters: [Sage Dirichlet-character reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html).
