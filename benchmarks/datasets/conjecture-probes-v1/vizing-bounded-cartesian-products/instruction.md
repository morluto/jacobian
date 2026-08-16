# Probe the Vizing domination-number Cartesian-product lower bound

For each of the eight frozen graphs `P4`, `C4`, `P5`, `C5`, `K2,3`, `house`,
`bull`, and `corona-K3` (adjacency lists in
`/app/input.json`), report the vertex count, edge count, and exact domination
number. For each of the thirteen frozen Cartesian pairs, report the product
vertex count, the exact domination number of the Cartesian product, the product
of the two factor domination numbers, and whether the Vizing lower bound
`gamma(G square H) >= gamma(G) * gamma(H)` holds for that pair. Also report a derived conclusion of exactly `HOLDS_ON_FROZEN_PAIR_SET` or
`VIOLATION_IN_FROZEN_PAIR_SET`.

The domination number of a graph is the minimum cardinality of a set of
vertices whose closed neighborhoods cover every vertex. The Cartesian product
`G square H` has vertex set `V(G) x V(H)`; two vertices `(g, h)` and
`(g', h')` are adjacent when `g = g'` and `h` is adjacent to `h'`, or `h = h'`
and `g` is adjacent to `g'`.

Submit `submission.json` including minimum dominating-set witnesses for both
factors and every product. The verifier reconstructs every domination number
and bound independently from frozen input using only the Python standard
library. This checks a finite family, not the open Vizing conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
