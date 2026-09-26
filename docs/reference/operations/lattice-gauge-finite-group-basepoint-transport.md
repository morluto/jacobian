# Finite-group loop basepoint transport

`lattice_gauge.finite_group.basepoint_transport.compute` accepts an exact
finite-table gauge field, a loop based at `s`, and a connector path `gamma`
from `s` to `t`. It returns the loop at `t` with path

\[
\gamma^{-1}\,\ell\,\gamma
\]

and the exact source-loop, connector, and transported-loop holonomies. With
the published left-to-right path product convention, the finite-group value
satisfies

\[
\operatorname{Hol}(\gamma^{-1}\ell\gamma)
=\operatorname{Hol}(\gamma)^{-1}\operatorname{Hol}(\ell)\operatorname{Hol}(\gamma).
\]

This is the finite groupoid version of the standard change-of-basepoint map
for fundamental groups: Hatcher, *Algebraic Topology*, §1.1, Proposition 1.5
([Chapter 1 PDF](https://pi.math.cornell.edu/~hatcher/AT/ATch1.pdf)).

The result retains the field, source loop, connector, transported path, both
basepoints, and all three parent-bound group elements. Thus a consumer can
feed the transported path unchanged to
`lattice_gauge.finite_group.holonomy.compute` and compare the exact output.
An empty connector at `s` is the identity transport; an empty source loop has
identity holonomy.

Admission bounds the finite group table, edge field, both source paths, the
derived path length `2*len(gamma)+len(ell)`, multiplication work, and output
units before constructing the transported path or multiplying holonomies.
This computes supplied finite data only; it does not solve gauge equivalence,
define continuum physics, or introduce a generic gauge formula language.
