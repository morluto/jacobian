# Finite-field irreducibility proof repair

This Regression benchmark transforms Xerv-AI/GRAD train row 2 at immutable
revision `71595210590450202b7b69225bc07e9e01b13c5c` (MIT). The source proof
incorrectly claims that `x^4+1` is irreducible over `F_2`.

The agent must diagnose the repeated-factor reduction and replace it with an
independently checkable finite-field irreducibility certificate for
`x^4-4x+1`. The verifier implements polynomial arithmetic over finite fields,
checks the bad factorization, and replays Rabin's degree-four criterion for a
freely chosen repair prime.

Family: **Regression**. Primary objective: **diagnose and repair a finite-field
irreducibility argument**. Difficulty: **Hard (provisional)** because it needs
proof auditing, prime selection, modular exponentiation, and exact certificate
construction. The ceiling is `COMPUTED`; the local irreducibility repair does
not verify the source proof's later Galois or density claims.

Shortcut audit: the public answer, the label “irreducible,” or the known prime
11 alone fails. Multiple repair primes can pass, and every submitted polynomial
remainder is recomputed.
