# Irreducible vertices in positive-definite weighted tree lattices

Let T be a finite tree with integer vertex weights w. Let L(T) be the free abelian group on V(T) with bilinear form u·u=w(u), u·v=-1 for adjacent distinct vertices, and u·v=0 otherwise. Assume the form is positive definite and that there exists exactly one vertex v with w(v)<d(v). Prove that some vertex u is irreducible in L(T): there are no nonzero a,b in L(T) with u=a+b and a·b>=0.

This is a public answer-visible diagnostic: the expected conclusion and oracle summary are
public. Reproduce the answer-visible conclusion, state the relevant answer-visible facts, and
report the operation boundary honestly. There is no domain-owned weighted-tree lattice artifact, irreducibility predicate, decomposition search, or general proof-producing checker. Generic matrix calculations do not expose the theorem's semantic objects.

Write `submission.json` to the exact agent-visible `submission_schema.json`, record your
reasoning in `evidence/answer.txt`, and include that file's SHA-256 digest in the evidence
list. In `result.key_facts`, use a nonempty object with lower-snake-case fact
names and nonempty string values; do not rely on prose or numeric/boolean
coercion. You may claim `UNVERIFIED` or `COMPUTED`; only `COMPUTED`
receives full aggregate credit. Do not claim `VERIFIED`. Treat timeout,
error, or incomplete search as a non-conclusion, not as evidence for or
against the claim.
