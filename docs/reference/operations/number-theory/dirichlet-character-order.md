# Exact Dirichlet-character order

`dirichlet_character.order.compute` returns the multiplicative order of one
character in the finite dual group. For a cyclic coordinate axis of order
`m`, coordinate `c` has order `m / gcd(c,m)`; the character order is the least
common multiple of these axis orders. The trivial unit group has one character
of order one.

The exact result retains the full source `DirichletCharacter`, including its
modulus and canonical unit-group coordinates. Its order divides the group
exponent, so no character values need to be expanded or approximated. Sage's
Dirichlet-character API exposes the same multiplicative-order operation and
examples: [Sage Dirichlet-character reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html).
