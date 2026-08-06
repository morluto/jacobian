# Closed one-form polynomial classification

Regression benchmark for chain-rule proof auditing, symbolic integrability, exact linear-algebra classification, and reconstructible potential certificates. The frozen GRAD solution incorrectly replaces the derivative of `f(y,x)` with `f_x(y,x)`; the correct derivative with respect to `x` is `f_y(y,x)`. The source problem is frozen from Xerv-AI/GRAD (MIT), train row 23, revision `71595210590450202b7b69225bc07e9e01b13c5c`.

The candidate scored 90/100. It adds a local proof-defect diagnosis plus a corrected coefficient-space classification workflow; nearby answer-only dimension questions were rejected. The shortcut audit rejects copying the published but incorrect dimension, dimension-only answers, fixed-basis matching, dependent bases, and potentials not bound to their basis elements. Difficulty is provisional Hard because the agent must detect the chain-rule error, connect global line integrals to closedness, derive the corrected constraint space, and produce independently replayable primitives.

The verifier establishes exact finite polynomial identities and linear-algebra facts. It does not machine-prove the analytic equivalence between vanishing closed-curve integrals and exactness on `R2`, so the ceiling is `COMPUTED`.
