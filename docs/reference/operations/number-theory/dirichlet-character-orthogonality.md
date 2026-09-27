# Exact Dirichlet-character orthogonality sum

`dirichlet_character.orthogonality.compute` returns the exact integer

\[
\sum_{a\bmod N}\chi(a)\overline{\psi(a)}.
\]

Both characters must use the identical complete `DirichletCharacterGroup`
parent. Values at nonunits are zero. The returned integer retains both source
characters. It is \(\varphi(N)\) when the characters are equal and zero when
they are distinct; the operation returns this scalar sum, not an equality
predicate.

The kernel uses the validated unit-coordinate bijection. If the group axes
have orders \(o_i\) and the source characters have coordinates \(c_i,d_i\),
the residue sum factors exactly as

\[
\prod_i\left(\sum_{j=0}^{o_i-1}
  \exp\left(2\pi i(c_i-d_i)j/o_i\right)\right).
\]

Each factor is \(o_i\) when \(c_i=d_i\pmod{o_i}\), and zero otherwise.
For the trivial unit group the empty product is one, giving the correct
modulus-one result.

Before evaluating the pairing, the operation validates each complete group
parent and admits its modulus, unit count, and coordinate work. The current
group modulus bound is 2,048; the coordinate work bound is 500,000 units. The
result is a strict exact integer with absolute value at most \(\varphi(N)\).
