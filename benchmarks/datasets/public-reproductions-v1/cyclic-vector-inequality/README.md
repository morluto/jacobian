# Cyclic vector inequality

This Regression benchmark transforms AI4Math/IneqMath dev row 12 at immutable revision `3c7c32c786eb77117f3476d7f6d9af8419fa6ecc` (CC-BY-SA-4.0). The public row contains the optimal constant but no solution.

The task requires a symbolic Minkowski reduction, coefficient-level cyclic cancellation, a completed-square nonnegativity certificate, and a sharp equality witness. The verifier accepts any supported certificate dimension and reconstructs all affine and quadratic coefficients independently.

Family: **Regression**. Primary objective: **symbolic sharp-inequality certification**. Quality score: **87/100**. Difficulty: **Hard (provisional)**; the proof chain is multi-stage but uses elementary trusted norm facts, so baseline calibration may lower it to Medium-Hard. The assurance ceiling is `COMPUTED` because the triangle inequality and norm monotonicity are trusted lemmas rather than proof-assistant replay.

Shortcut audit: the published constant, a decimal approximation, sampled inputs, or an equality case without the universal lower-bound certificate cannot pass. Nearby IneqMath rows were rejected when only direct computation or a fixed answer could be checked.
