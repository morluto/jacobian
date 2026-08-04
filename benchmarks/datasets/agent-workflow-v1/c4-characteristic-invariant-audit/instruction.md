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
conjecture is proved. Write the exact schema to `submission.json`. Put a concise
audit in `evidence/answer.txt`, include a `RESULT_JSON:` line containing the
submitted result as JSON, and bind its SHA-256 digest.
