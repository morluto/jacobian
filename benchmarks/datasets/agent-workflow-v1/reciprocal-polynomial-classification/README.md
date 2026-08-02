# Reciprocal-polynomial functional-equation certificate

Discover a nontrivial polynomial satisfying
`1/P(z) + 1/P(1/z) = z + 1/z`, and expose exact degree, quotient, reversal, and
Laurent-identity evidence. The agent-visible prompt does not disclose the
alternating coefficient family.

This Regression benchmark is derived from `lm-provers/FineProofs-SFT` train row
376 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f`
(Apache-2.0). The public answer is not trusted: the verifier rebuilds the
sparse polynomial, its reversal, the geometric quotient, and both sides of the
cleared Laurent-polynomial identity with exact integer arithmetic.

## Quality and shortcut audit

Quality score: **87/100**. The single primary objective is exact symbolic
functional-equation certification. Difficulty is **Hard (provisional)**: the
agent must derive an alternating odd-power family, preserve degree and reversal
conventions, and produce a non-tiny member; empirical calibration is pending.
The family index is chosen by the agent in `[6,20]`, and the family formula is
not supplied, so a memorized smallest solution or numerical sampling cannot
pass.

The verifier establishes the submitted family member and the algebraic
relations in the frozen classification chain. It trusts the general
degree/divisibility argument that every solution enters this family, and caps
assurance at `COMPUTED`; this public regression is not held-out evidence.
