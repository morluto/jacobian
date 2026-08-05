# Audit two incompatible C4 invariants

The frozen source uses `countInducedC4(G)` where the historical notation means
the C4-free characteristic: 1 if the graph has no four-cycle as a subgraph and
0 otherwise. A four-cycle may have chords.

Submit three connected finite simple graph certificates:

1. a C4-free graph with zero induced four-cycles and characteristic 1;
2. a graph with at least two induced four-cycles and characteristic 0;
3. a graph with a chorded four-cycle, no induced four-cycle, and characteristic
   0.

Graphs use vertices `0..vertex_count-1`; every edge is a strictly increasing
pair and the edge list is lexicographically sorted. Respect each witness's
vertex bounds. Report the exact induced-C4 count and characteristic value.

Do not claim the upstream Lean theorem compiles or that the source-corrected
conjecture is proved. State both limitations in the `limitations` field. Write
the exact schema to `submission.json`. Put a concise audit in
`evidence/answer.txt`, include a `RESULT_JSON:` line containing the submitted
result as JSON, and bind its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, exact finite scope, completeness, digest-bound evidence, limitation claims, and assurance as separate protocol dimensions. State both that Lean compilation is not assessed and that proof of the source-corrected conjecture is not claimed.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `INVARIANTS_NOT_EQUIVALENT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
