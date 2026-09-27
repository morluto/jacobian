# Exact generalized Bernoulli number for a Dirichlet character

`dirichlet_character.generalized_bernoulli.compute` returns

\[
B_{k,\chi}=N^{k-1}\sum_{a=1}^{N}\chi(a)B_k(a/N),\qquad k\ge0,
\]

where the classical Bernoulli polynomials use the generating function
\(te^{xt}/(e^t-1)\), so \(B_1(x)=x-\tfrac12\). This matches the exact
Dirichlet-character convention documented by [Sage](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html#sage.modular.dirichlet.DirichletCharacter.bernoulli).

The endpoint at \(a=N\) is included. Thus for the trivial character modulo
one, \(B_{1,\chi}=B_1(1)=+\tfrac12\), whereas the classical Bernoulli number
is \(B_1=-\tfrac12\). At index zero the factor is \(N^{-1}\), giving
\(B_{0,\chi}=N^{-1}\sum_{a=1}^N\chi(a)\); for example, a principal character
modulo 3 gives \(2/3\).

`dirichlet_character.generalized_bernoulli_prefix.compute` returns
\((B_{0,\chi},B_{1,\chi},\ldots,B_{K,\chi})\) through a requested maximum
index (K\), including both endpoints. Its result carries the source character
and maximum index, preserving the character interpretation and canonical
value field of every entry.

The exact result carries its source character and index and lies in the
canonical field \(\mathbb Q[\zeta_d]\), where \(d\) is the multiplicative
order of that character; \(d=1\) denotes \(\mathbb Q\). Character values use
the positive root \(\zeta_d=e^{2\pi i/d}\), and coefficients are reduced in
the power basis modulo \(\Phi_d\).

Admission bounds \(k\le32\), the source-group/unit-coordinate work, field order
\(d\le128\), coefficient digits, exact work (500,000 units), and estimated
serialized result bytes (1,000,000) before evaluating the finite sum. For the
coefficient preflight, \(B_j\) has denominator dividing \((j+1)!\) and
\(|B_j|\le2j!\); hence the coefficient l1 norm of \(B_k(x)\) is less than
\(6k!\). The exact cyclotomic polynomial's coefficient l1 norm then bounds
the admitted power-basis reduction. Requests outside any bound are rejected
without returning a partial value. The prefix operation admits aggregate work
and output bytes for all (K+1) values and preflights coefficient growth at
every index before evaluating any value.
