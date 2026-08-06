# Möbius-orbit functional-equation classification

This Regression benchmark is derived from PutnamBench
`putnam_1971_b2` at immutable revision
`dfb0a47a1c1ec3a10f2a9acfdf41a2043920f33c` (the Lean source is
Apache-2.0; the informal statement is distributed by PutnamBench with MAA
permission). The public answer is not trusted. The verifier replays the
period-three transform, rational right-hand sides, all three functional
equations, and the nonsingular orbit linear system.

## Quality and shortcut audit

Quality score: **90/100**. The single primary objective is exact functional
equation classification. Difficulty is **Hard (provisional)**: the agent must
discover the finite Möbius orbit, maintain denominator domains, solve a
rational-function linear system, and prove uniqueness; empirical calibration
is pending. Memorizing the published formula does not supply the two other
orbit values or the independently replayed cycle and matrix certificate.

The verifier establishes exact rational identities and uniqueness on every
declared three-cycle. It trusts the standard set-theoretic step that these
cycles cover the frozen domain and caps assurance at `COMPUTED`.
