# Exact Dirichlet-character Jacobi sum

`dirichlet_character.jacobi_sum.compute` returns

\[
J(\chi,\psi)=\sum_{a\in\mathbb Z/N\mathbb Z}\chi(a)\psi(1-a).
\]

This is the standard Jacobi-sum convention used by Sage's exact Dirichlet
character API ([definition and examples](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html#sage.modular.dirichlet.DirichletCharacter.jacobi_sum)).
Character values at nonunits are zero. Both input characters must carry the
same complete `DirichletCharacterGroup` parent.

The output is a `RationalCyclotomicElement` in the canonical parent
\(\mathbb Q[\zeta_e]=\mathbb Q[x]/(\Phi_e(x))\), where \(e\) is that group's
common character-value order and \(x\) denotes the class of
\(\exp(2\pi i/e)\). The output stores exactly \(\varphi(e)\) rational
coefficients in ascending power-basis order. This gives an explicit embedding
for every character value and a stable field identity for downstream exact
operations.

The operation admits modulus at most 2,048, cyclotomic order at most 128, at
most 1,000,000 residue-and-field work units, and the shared 256-digit
cyclotomic coefficient limit. It admits the field construction before forming
its defining polynomial and proves a coefficient-growth bound from the exact
cyclotomic polynomial before enumerating the sum. Requests whose field or
reduced coefficients exceed these limits are rejected in full.
