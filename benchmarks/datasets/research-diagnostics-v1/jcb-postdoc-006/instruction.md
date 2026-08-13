# Written on the Wall II Conjecture 59 at portfolio scale limits

Let B have bipartition X={x_0,...,x_11}, Y={y_0,...,y_11}, with x_i adjacent to y_j iff (j-i) mod 12 belongs to {0,1,2,3,4,5,6}. Let H be the disjoint union of B and 98 isolated vertices, and let G be the cone over H. Here r(G) is the Havel-Hakimi residue, b(G) the maximum order of an induced bipartite subgraph, and f(G) the maximum order of an induced forest. Prove r(G)=101, b(G)=122, f(G)=111, and hence f(G)<ceil(sqrt(r(G)b(G))).

This is a public answer-visible diagnostic: the expected conclusion and oracle summary are
public. Reproduce the answer-visible conclusion, state the relevant answer-visible facts, and
report the operation boundary honestly. Current graph contracts have bounded vertex envelopes below the full 123-vertex instance. The decisive proof must use visible structural lemmas, not pretend a rejected or truncated invocation certified the full graph.

Write `submission.json` to the exact agent-visible `submission_schema.json`, record your
reasoning in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence
list. In `result.key_facts`, use a nonempty object with lower-snake-case fact
names and nonempty string values; do not rely on prose or numeric/boolean
coercion. You may claim `UNVERIFIED` or `COMPUTED`; only `COMPUTED`
receives full aggregate credit. Do not claim `VERIFIED`. Treat timeout,
error, or incomplete search as a non-conclusion, not as evidence for or
against the claim.
