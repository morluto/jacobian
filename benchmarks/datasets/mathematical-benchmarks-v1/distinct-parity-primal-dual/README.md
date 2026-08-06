# Distinct-parity primal/dual certificate

Regression benchmark derived from ByteDance-Seed/BeyondAIME test row 38 at immutable revision `c705198ae1043810b1e1693bd879250b51a7a523` (CC0-1.0). It adds an exact optimization workflow with separate primal feasibility and upper-bound frontier evidence.

Family: Regression. Primary objective: discrete optimization. Quality score: 87/100. Difficulty: Hard (provisional): a successful agent must construct an attaining set and derive a parity-complete frontier rather than only guess the published optimum.

Shortcut audit: the answer `384` alone fails, as do non-distinct values, wrong parity, or an incomplete frontier. The verifier accepts alternative optimal constructions and independently reconstructs all 23 upper-bound rows. Assurance remains `COMPUTED` because no proof assistant replays the elementary minimum-sum argument.
