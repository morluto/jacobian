# Gram-Schmidt nonzero-filter audit

An Assurance benchmark derived from FormalRx-Test row 17. It isolates a
boundary-predicate mistranslation: `‖u‖ ≥ 0` selects every residual, including
zero, whereas the informal theorem removes zero residuals.

The task was selected over nearby one-token theorem-name repairs because a
valid response must construct a dense rank-four system in `Q^5` and replay six
exact Gram-Schmidt residuals. Standard-coordinate tiny witnesses are excluded,
and alternate systems are accepted. Primary objective: semantic alignment.

Provisional difficulty is **Hard**, quality score **86/100**. Weaker agents are
expected to notice nonnegativity but often fail exact residual or rank binding;
stronger agents can construct and certify a dependent dense system. Trust ends
at exact rational linear algebra. Lean elaboration and the theorem outside this
finite countermodel are not assessed.
