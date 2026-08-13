# Exact Farkas-certificate slice audit

This Assurance benchmark is derived from the canonical exact rational
certificate in `Oxelra-AI/Bandeira-0.2a-open-source` at immutable revision
`c41373447abfadabc30afeebc3c4de74e1ccf8dc` (MIT). The 4×4 input is the
top-left principal submatrix of block 1 of `-M`, reconstructed from the
source certificate and standard-form builder. Historical floating-point SDP
outputs are explicitly not treated as proof evidence.

## Quality and shortcut audit

Quality score: **92/100**. The primary objective is exact certificate replay
with calibrated scope. Difficulty is **Hard (provisional)**: the agent must
perform large exact rational linear algebra, preserve artifact semantics, and
avoid promoting a local check to the full research claim; empirical
calibration is pending. Either an exact LDL factorization or a Sylvester
certificate is accepted. Floating-point eigenvalues, copied source status, or
a single positive minor cannot pass.

The verifier independently checks reduced fractions, the published scalar
replay `m00 = y0 + c00_y` with `m00 < 0` and `objective > 0`, and the selected
positive-definiteness certificate. It intentionally does not check the
remaining matrix entries, the other five blocks, the full Farkas implication,
or Lean, and caps assurance at `COMPUTED`.
