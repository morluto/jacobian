# Chebotarev fixed-point proof audit

Regression benchmark derived from Xerv-AI/GRAD train row 2 at immutable revision `71595210590450202b7b69225bc07e9e01b13c5c` (MIT). It requires a multi-stage audit of a published research-level solution rather than answer recovery.

Family: Regression. Primary objective: proof diagnosis. Quality score: 92/100. Difficulty: Hard (provisional): the task separates polynomial arithmetic, finite-field factorization, permutation conjugacy classes, and the conditional Chebotarev conclusion.

Shortcut audit: returning `304` or correcting only one error fails. The verifier independently expands the mod-2 factor, computes the quartic discriminant as a Sylvester resultant, enumerates all 24 permutations, and checks every class. The Galois-group classification and Chebotarev theorem are explicit trusted boundaries, so assurance is `COMPUTED`.
