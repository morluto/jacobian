# Dirichlet-character orthogonality over the full dual group

`dirichlet_character.orthogonality_over_characters.compute` returns the exact
sum

```text
sum over chi mod N of chi(a) * conjugate(chi(b))
```

for the complete finite Dirichlet-character group. Characters use the usual
extension by zero on nonunits. Therefore the result is `phi(N)` when both
residues are units and equal modulo `N`, and zero in every other case. This is
the column orthogonality relation for the character table; the equivalent
finite-abelian-group identity is documented by
[Mathlib's Dirichlet-character orthogonality theorem](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/DirichletCharacter/Orthogonality.html).

The input includes a complete typed `DirichletCharacterGroup` and two bounded
integers. The output preserves that group and both source integers, alongside
their canonical residues and the exact integer sum. Modulus one is handled by
its unique residue class `0`, which is a unit modulo one. Nonunit residues have
character value zero by definition.

The kernel applies the finite dual-group formula directly after validating the
group decomposition. It does not build a character family or materialize
roots of unity. The supplied group still carries the exact common
root-of-unity exponent; its modulus, unit table, coordinate axes, work, and
serialized result are bounded before residue normalization. This operation
does not return individual character values or a cyclotomic sum.

The distinct operation
[`dirichlet_character.orthogonality.compute`](dirichlet-character-orthogonality.md)
sums over residues for one fixed character pair; this operation sums over all
characters for one residue pair.
