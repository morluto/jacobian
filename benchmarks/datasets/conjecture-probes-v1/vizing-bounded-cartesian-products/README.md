# Vizing bounded Cartesian-product domination probe

This **authored conjecture-probe** benchmark fixes eight small adjacency-list
graphs and thirteen Cartesian pairs and asks for the exact domination number of
each factor, each Cartesian product, and whether the Vizing lower bound
`gamma(G square H) >= gamma(G) * gamma(H)` holds for every pair. Its single
primary objective is bounded exact graph-theoretic recomputation over products
of at most 30 vertices.

- **Portfolio contribution:** finite domination-number enumeration and
  Cartesian-product construction over frozen adjacency-list graphs, distinct
  from orbit enumeration and fixed proof templates.
- **Shortcut audit:** the aggregate `all_bounds_hold` flag is insufficient;
  every factor domination number, product domination number, factor product,
  and per-pair bound check must agree with an independent recomputation.
- **Difficulty:** Medium, provisional. Correctness requires exact domination
  enumeration over products of up to 30 vertices and consistent per-pair
  bound checks.
- **Discrimination:** weak agents tend to report only the aggregate flag or
  omit the factor domination numbers; stronger agents produce a complete
  per-graph and per-pair certificate.
- **Quality score:** 80/100.

The verifier independently recomputes every domination number and bound from
the frozen input using only the Python standard library. It accepts alternate
minimum dominating sets, binds one JSON certificate byte-for-byte to the
visible evidence path, and reports protocol, input binding, mathematics,
evidence, scope, assurance, and aggregate reward separately. A finite probe is
computational evidence, never a proof of the open Vizing conjecture.

## Curation and sources

Spreadsheet row C-013 was a curation lead only and is not executable or
authoritative. The input was frozen on 2026-08-06 after consulting the
[Brešar–Dorbec–Goddard–Hartnell–Henning–Klavžar–Rall survey](https://doi.org/10.1002/jgt.20565),
the computational study
[Towards a computational proof of Vizing's conjecture](https://doi.org/10.1016/j.jsc.2021.01.003),
and Steiner's current progress paper
[A constant-factor step towards Vizing's conjecture](https://arxiv.org/abs/2606.14414).
Those sources motivate the bounded probe; none supplies the finite certificate
or authorizes a global conclusion.
