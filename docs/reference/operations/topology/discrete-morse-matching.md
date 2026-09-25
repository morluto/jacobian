# Discrete Morse matching

[Topology operations](index.md) · [Tool surface](../../tools.md)

`topology.discrete_morse.matching.minimum.compute` takes a finite simplicial
complex and returns an acyclic matching with the minimum **total** number of
critical cells. For a matching with `m` pairs on a complex with `n` nonempty
faces, the critical-cell total is `n - 2m`; therefore this operation maximizes
the number of matched cover pairs. It does not minimize a dimension-wise
objective or claim that the Morse complex is a simplicial subcomplex.

The operation exhaustively considers matchings in canonical face-poset cover
order and accepts only acyclic directed Hasse graphs. Equal optima are resolved
by the first optimum found by include-first traversal of sorted cover pairs.
The returned value contains the source-bound matching and its complete
critical-cell profile.

Optimal Morse matching is NP-hard in general. The search branches on each
codimension-one incidence in canonical order and includes a cover only when
both of its cells are still unmatched, so its reachable states are partial
matchings rather than all `2^E` subsets. Jacobian counts the nodes of that
disjointness-pruned search tree and admits a request only when the resulting
work estimate is at most 20,000,000 units, charging each node at most
`N + E + 1` linear work units, where `N` is the number of nonempty faces and
`E` the number of codimension-one incidences. A cheap `(E + 1)(E + 2) / 2`
lower bound on the node count refuses oversized searches before the exact
count runs. It also bounds the exact result by an 8,000,000-byte output
envelope before search. An over-envelope request returns a resource-admission
error; no partial matching is labeled minimum. These limits bound this
implementation and do not change the mathematical optimization problem.

The objective and hardness distinction follow Joswig and Pfetsch, [*Computing
Optimal Morse Matchings*](https://arxiv.org/abs/math/0408331). Their paper
identifies maximum-cardinality Morse matching with minimizing the number of
critical faces and proves the general problem NP-hard.
