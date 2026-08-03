# Written on the Wall II Conjecture 194: discover a finite obstruction

For a connected finite simple graph G, let alpha(G) be its independence number
and let l_avg(G) be the average of alpha(G[N(v)]) over all open neighborhoods.
Disprove the assertion that alpha(G) <= 1 + l_avg(G) forces G to have a
Hamiltonian path. Produce an explicit counterexample with at most 20 vertices.

Write the graph and all claimed invariants to
`evidence/counterexample.json` using the exact agent-visible
`certificate_schema.json`. Edges must use consecutive integer vertex labels,
be sorted pairs `u < v`, and appear in lexicographic order. The local average
must be a reduced nonnegative rational. Bind that file in `submission.json`
using its SHA-256 digest and the exact `submission_schema.json` contract.

This is a public answer-visible diagnostic. The verifier independently checks
connectivity, every independence number, the exact inequality, and absence of
a Hamiltonian path. Claim `COMPUTED` only. Timeout, malformed evidence, or an
incomplete calculation is a non-conclusion, never evidence for the claim.
