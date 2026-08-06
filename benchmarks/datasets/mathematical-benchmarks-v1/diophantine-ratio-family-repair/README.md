# Diophantine ratio-family proof repair

Hard provisional Regression benchmark from Xerv-AI/GRAD (MIT), train row 19, revision `71595210590450202b7b69225bc07e9e01b13c5c`. Score: 91/100.

The frozen proof's reverse-Vieta partner is nonintegral, invalidating its entire recurrence. The task must both diagnose that exact failure and replace it with an integer-polynomial family whose divisibility identity is independently replayed over `Z`. Free probe parameters are accepted, so a fixed numeric trace is insufficient.

This adds constructive repair of an infinite Diophantine family, distinct from finite counterexample search, bounded enumeration, Galois certificates, or local proof-label replay. The verifier proves the submitted polynomial identities and exact instances, while making no classification claim beyond the family; assurance is capped at `COMPUTED`.
