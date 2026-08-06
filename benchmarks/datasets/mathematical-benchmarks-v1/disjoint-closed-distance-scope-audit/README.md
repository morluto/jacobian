# Disjoint closed-set distance scope audit

This Assurance benchmark is derived from ProofNetVerif test row 18 at immutable revision `91183e5b12d64374827bf2782db629b5b0f8f319` (MIT). The row replaces topological separation of two closed sets with a stronger uniform positive-distance assertion.

The agent must produce a non-tiny parametric countermodel in `R^2`, exact rational distance certificates, eight independently checked epsilon witnesses, and a closedness/separation certificate. The verifier accepts many parameter and witness choices and rejects finite-only, intersecting, nonclosed, malformed, or falsely VERIFIED submissions.

Family: **Assurance**. Primary objective: **semantic scope audit**. Difficulty: **Hard (provisional)** because the task coordinates a general metric countermodel, exact rational bounds, and topology/metric scope separation; empirical calibration is pending. Quality score: **86/100**.

Assurance is capped at `COMPUTED`: exact algebra and the frozen parametric family are replayed independently, while Lean elaboration and a foundational formal proof of closedness are not assessed.
