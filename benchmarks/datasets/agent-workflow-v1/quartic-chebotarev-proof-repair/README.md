# Quartic Chebotarev proof repair

Hard provisional Regression benchmark from Xerv-AI/GRAD (MIT), train row 2, revision `71595210590450202b7b69225bc07e9e01b13c5c`. Score: 92/100.

The public solution contains four coupled defects: a reducible mod-2 polynomial is called irreducible, the discriminant is wrong, fixed-point-free double transpositions are counted as fixed-point elements, and the resulting density is wrong. The task requires a replacement irreducibility certificate, exact discriminant, independently checkable Frobenius factorization, subgroup argument, and complete `S4` action count. Several unramified factorization witnesses are accepted, so the benchmark does not require one memorized trace.

This adds Galois/Chebotarev proof repair rather than another bounded search or fixed proof-label replay. The checker is exact on polynomial arithmetic and the finite group action, with a `COMPUTED` ceiling because the named global theorems are trusted rather than formalized.
