# Square-free affine infinite product enclosure

`number_theory.squarefree_affine_forms.infinite_product.enclose` returns an
exact rational interval containing the convergent product of local square-free
densities for a finite family of integer affine forms.

For each prime (p), let α_p be the proportion of residues modulo (p^2) on
which none of the supplied forms is divisible by (p^2). Given a cutoff (P),
the operation computes every factor for (p \le P) exactly. The cutoff must be
at least

\[
B=\max\left(\max_j |a_j|,\left\lfloor\sqrt{M}\right\rfloor\right),
\qquad M=\max\bigl(\#\{L_j\}, |b_j|:a_j=0, 1\bigr).
\]

For (p>P\ge B), every nonconstant form has exactly one bad residue modulo
(p^2), while a nonzero constant form has none. Therefore

\[
0\le 1-\alpha_p\le \frac{k}{p^2},
\qquad
\sum_{p>P}(1-\alpha_p)
\le k\sum_{n>P}\frac1{n^2}
\le \frac{k}{P},
\]

where (k) is the number of forms. Since the factors lie in ([0,1]), the
omitted product lies in \([\max(0,1-k/P),1]\). If
\(A_P=\prod_{p\le P}\alpha_p\), the returned rational interval is
\([A_P\max(0,1-k/P),A_P]\). All endpoint arithmetic is exact rational
arithmetic; no floating-point approximation is involved.

The cutoff is at most 1000 and aggregate local-factor work is admitted from
\(k\sum_{p\le P}p^2\). The prefix has fewer than 1000 factors, each with
denominator at most (10^6), so the rational endpoints stay below the
interoperable rational digit ceiling. A local obstruction in the checked
prefix makes the infinite product exactly zero. The operation makes no claim
that the family has simultaneous square-free integer values, or that such
values have a positive density.
