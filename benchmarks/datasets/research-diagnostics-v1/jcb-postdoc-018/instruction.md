# WOWII Conjecture 18: detect a semantic mismatch between two graph invariants

Let G have vertices {0,1,2,3,4,5} and edges {03,04,14,24,34,35}. Let M be the maximum-degree vertices. Define dist_max(M)=max{dist(u,v):u,v in M}, with value 0 for a singleton, and eccSet(M)=max{dist(v,M):v in V}. Let alpha(G) be the independence number and b(G) the maximum order of an induced bipartite subgraph. Compute M, alpha(G), b(G), dist_max(M), and eccSet(M). Decide the truth of alpha(G)+ceil(sqrt(eccSet(M))) <= b(G) and alpha(G)+ceil(sqrt(dist_max(M))) <= b(G).

This is a public answer-visible diagnostic: the expected conclusion and oracle summary are
public. Reproduce the answer-visible conclusion, state the relevant answer-visible facts, and
report the operation boundary honestly. There is no dedicated set-eccentricity or dist_max-on-a-designated-set operation. A graph radius or diameter result must not be relabeled as either requested invariant.

Write `submission.json` to the exact agent-visible `submission_schema.json`, record your
reasoning in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence
list. You may claim `UNVERIFIED` or `COMPUTED`; only `COMPUTED` receives
full aggregate credit. Do not claim `VERIFIED`. Treat timeout, error, or
incomplete search as a non-conclusion, not as evidence for or against the claim.
