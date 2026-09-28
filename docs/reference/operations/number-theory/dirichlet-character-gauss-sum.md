# Exact Dirichlet character Gauss sum

`dirichlet_character.gauss_sum.compute` returns

\[
\tau(\chi)=\sum_{a\bmod N}\chi(a)\exp(2\pi i a/N)
\]

for one exact `DirichletCharacter`. Character values use extension by zero off
units. The result retains its source character and is represented canonically
in `QQ[zeta_m]`, where `m = lcm(ord(chi), N)` and `ord(chi)` is the actual
order of the character's values. No primitivity hypothesis is imposed.

The operation admits the target order, residue work, cyclotomic reduction work,
and a conservative exact coefficient bound before constructing the target
cyclotomic polynomial or summing. Requests outside the admitted envelope return
a resource-admission error rather than an incomplete value.

`dirichlet_character.generalized_gauss_sum.compute` returns the same exact
sum at an explicit signed integer frequency:

\[
\tau_n(\chi)=\sum_{a\bmod N}\chi(a)\exp(2\pi i n a/N).
\]

The result retains the authored frequency and its canonical residue modulo
`N`. Zero and nonunit frequencies are included. For example, the zero-frequency
sum of the quadratic character modulo 5 is zero, while the principal character
sum is 4. The ordinary Gauss-sum operation is the frequency-one case.

`dirichlet_character.primitive_gauss_norm.compute` consumes a
`PrimitiveDirichletCharacter` carrier with a claimed exact conductor. It checks
that claim against the computed conductor before using the norm identity. It
returns the exact cyclotomic Gauss sum and its product with its cyclotomic
conjugate, checking
`tau(chi) * conjugate(tau(chi)) = N` by exact reduction. This is the squared
complex absolute value for a primitive character; imprimitive characters are
rejected rather than assigned the primitive norm formula.
