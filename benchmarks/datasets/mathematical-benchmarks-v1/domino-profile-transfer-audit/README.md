# Domino profile-transfer audit

This **Regression** benchmark is derived from `INSAIT-Institute/OPC` test row
5 at immutable revision `dcc3b4804e2d126ea34b13e3e0cd998c3302644b`
(Apache-2.0). The frozen solution receives score zero and proposes remainder
4 after an unjustified decomposition and period claim.

The agent must replace that argument with a complete eight-state profile
transition matrix and a modular binary-exponentiation trace. The verifier
independently enumerates all legal domino fillings of one column, reconstructs
the matrix, replays every selected exponent bit, and checks the final empty
profile. Top-left and bottom-left missing corners are both accepted.

Primary reasoning objective: **repair an invalid enumerative proof with an
exact finite-state certificate**. Difficulty is **Hard (provisional)** because
the task requires discovering the correct state model and coordinating a long
modular computation; the certificate is then replayable in linear-algebraic
steps. The shortcut audit rejects a bare remainder, a copied recurrence, stale
period labels, and any corrupted transition or exponentiation state.

Assurance is capped at **COMPUTED**.
