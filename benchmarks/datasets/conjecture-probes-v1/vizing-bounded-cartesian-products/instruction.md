# Probe the Vizing domination-number Cartesian-product lower bound

For each of the eight frozen graphs `P4`, `C4`, `P5`, `C5`, `K2,3`, `house`,
`bull`, and `corona-K3` (adjacency lists in
`/app/input.json`), report the vertex count, edge count, and exact domination
number. For each of the thirteen frozen Cartesian pairs, report the product
vertex count, the exact domination number of the Cartesian product, the product
of the two factor domination numbers, and whether the Vizing lower bound
`gamma(G square H) >= gamma(G) * gamma(H)` holds for that pair. Also report
report a derived conclusion of exactly `HOLDS_ON_FROZEN_PAIR_SET` or
`VIOLATION_IN_FROZEN_PAIR_SET`.

The domination number of a graph is the minimum cardinality of a set of
vertices whose closed neighborhoods cover every vertex. The Cartesian product
`G square H` has vertex set `V(G) x V(H)`; two vertices `(g, h)` and
`(g', h')` are adjacent when `g = g'` and `h` is adjacent to `h'`, or `h = h'`
and `g` is adjacent to `g'`.

Submit `submission.json` and digest-bind `evidence/answer.txt`, which must be
a JSON object that copies `result` and `limitations` exactly and includes
`schema_version` set to `"1"` and `task_id` set to the `task_id` value from
`/app/input.json`. The verifier reconstructs every domination number and bound
independently from the frozen input using only the Python standard library.
Include minimum dominating-set witnesses for both factors and every product.
Claim `CHECKED`; the scope is the exact frozen graph and pair identity. This is
computational evidence for a finite family, not a proof of the open Vizing
conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The clean-room verifier reconstructs all frozen Cartesian products and independently recomputes domination numbers; the finite result does not imply a global Vizing conclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `VIZING_BOUNDED_PROBE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
