# Sharp Cauchy-composition inequality certificate

This Regression benchmark is derived from `AI4Math/IneqMath` train row 1 at
immutable revision `3c7c32c786eb77117f3476d7f6d9af8419fa6ecc`
(CC-BY-SA-4.0). The source answer is not trusted. The verifier reconstructs
the submitted sparse polynomials, replays one of two proof modes, and checks a
positive rational equality witness proving sharpness.

## Quality and shortcut audit

Quality score: **89/100**. The primary objective is exact sharp-constant
reasoning. Difficulty is **Hard (provisional)** because the agent must choose
and instantiate a symbolic proof architecture, preserve six-variable sparse
polynomial identities, and separately establish optimality; empirical
calibration is pending. Two structurally different certificates are accepted,
so the schema does not prescribe a unique proof trace. Sampling, a memorized
constant, or a tiny witness without the universal upper-bound certificate
cannot pass.

The verifier establishes the exact polynomial identities and sharpness
witness. It trusts the positive-homogeneity normalization plus elementary
order, square-root, AM-GM, and Cauchy--Schwarz lemmas and therefore caps
assurance at `COMPUTED`.
