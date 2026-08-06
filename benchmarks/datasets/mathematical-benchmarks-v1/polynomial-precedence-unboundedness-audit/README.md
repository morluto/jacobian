# Polynomial-precedence unboundedness audit

This Assurance benchmark is derived from FormalRx-Test row 2. The informal
expression squares `x*y`; the formal statement parses `x*y^2`. The task asks
for an independently replayable parametric certificate that the latter is
unbounded below, without exposing a preferred family.

## Curation

Selected because it tests precedence-sensitive semantic alignment, exact
symbolic substitution, and the distinction between a countermodel and Lean
elaboration. Nearby rows were rejected as theorem-name or constant edits with
tiny counterexamples. The primary objective is semantic audit; it does not test
proof synthesis. Family: **Assurance**.

Shortcut audit: a single low-valued point is insufficient. A submission must
give a polynomial family, its independently recomputed expansion, a negative
leading coefficient, and four exact checkpoints. Equivalent rational families
are accepted. The provisional **Hard** label reflects the need to discover and
certify an unbounded direction rather than merely notice different syntax;
empirical calibration may lower it to Medium–Hard.

Quality score: **87/100**. Trust is limited to exact rational polynomial
arithmetic in the clean-room verifier. The result is `COMPUTED`; Lean
elaboration and the truth of the informal claimed minimum are not assessed.
