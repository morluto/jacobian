# Finite Abelian character tables

`finite_abelian_group.character_table.compute` returns the full Fourier table
for an explicit product `G = C_m1 x ... x C_mr`. Both rows (dual frequencies)
and columns (group elements) use lexicographic coordinate order. For a row
`lambda`, column `x`, and exponent `E = lcm(m1,...,mr)`, the entry is

`zeta_E ^ sum_j(lambda_j * x_j * E/m_j)`.

Entries use the exact rational power basis of `Q(zeta_E)`. The table retains the
original product-group value, every coordinate, and the common cyclotomic
order, so consumers can use entries without converting through floating
complex numbers. The trivial group (empty product) has a one-by-one table.

Admission is based on the complete output: group order at most 128, exponent at
most 60, and at most 16,384 matrix cells / 655,360 exact coefficient cells.
This operation handles finite Abelian products only; it does not construct
characters of arbitrary finite groups or infer a decomposition from a group
table.
