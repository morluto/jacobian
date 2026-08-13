# Written on the Wall II Conjecture 200: a 14-vertex exact counterexample

Let G be obtained from K_{6,8}, with parts L={0,1,2,3,4,5} and R={6,7,8,9,10,11,12,13}, by deleting the twelve edges (0,6),(0,8),(1,8),(1,11),(2,7),(2,12),(3,6),(3,11),(4,7),(4,13),(5,12),(5,13). Let tree(G) be the maximum number of vertices in an induced tree and let lambda_avg(G) be the average, over vertices v, of the independence number of the open neighborhood G[N(v)]. Verify exactly that G is connected, tree(G)=ceil(1+lambda_avg(G))=7, and G has no Hamiltonian path.

This is a public answer-visible diagnostic: the expected conclusion and oracle summary are
public. Reproduce the answer-visible conclusion, state the relevant answer-visible facts, and
report the operation boundary honestly. The current portfolio has no dedicated Hamiltonian-path decider in the nearby set, so the alternation obstruction may remain an agent-owned proof step. Exact maximum output must expose completeness, not only a lower-bound witness.

Write `submission.json` to the exact agent-visible `submission_schema.json`, record your
reasoning in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence
list. In `result.key_facts`, report exactly the structured public facts
`tree: "7"`, `lambda_avg: "36/7"`, and `hamiltonian_path: "absent"`; these
are canonical strings, not inferred prose. You may claim `UNVERIFIED` or
`COMPUTED`; only `COMPUTED` receives full aggregate credit. Do not claim
`VERIFIED`. Treat timeout, error, or incomplete search as a non-conclusion,
not as evidence for or against the claim.
