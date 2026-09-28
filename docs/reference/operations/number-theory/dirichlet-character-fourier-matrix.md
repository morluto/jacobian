# Exact Dirichlet-character Fourier matrix

`dirichlet_character.group.fourier_matrix.compute` returns the complete finite
Fourier matrix for one admitted `DirichletCharacterGroup`. Its rows use every
dual character coordinate once in lexicographic order; columns use the group's
increasing canonical unit residues. The result retains the exact group parent
and both axes.

An entry `e` denotes `zeta_E^e` in the declared common cyclotomic parent of
order `E = group.exponent`. For a character coordinate
`c = (c_j)` and a unit's generator coordinate `u = (u_j)`,

```text
e = sum_j c_j u_j (E / order_j) mod E.
```

The matrix is the character restriction to units. Dirichlet characters remain
zero on nonunits by their existing extension convention; those residues are not
columns in this Fourier matrix. This agrees with the standard Dirichlet
character definition as a unit-group homomorphism extended by zero ([Sage
reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html)).

For the complete group of order `h = phi(N)`, character orthogonality gives
`F F* = h I` and `F* F = h I`, with conjugation in the same cyclotomic parent.
The operation returns the exact matrix itself, not a second copy of derived
Gram matrices. The identity is independently checked in tests by reducing
cyclotomic polynomials over the rationals.

The input group's defining decomposition is admitted before computation. The
full `h^2 * max(rank, 1)` cell arithmetic and a conservative encoded output
bound are then checked before any complete matrix coordinate or entry table is
built. Consequently, some already valid character groups are too large for a
complete matrix result.
