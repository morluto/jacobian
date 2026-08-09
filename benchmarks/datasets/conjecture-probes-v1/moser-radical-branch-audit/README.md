# Moser radical-branch assurance audit

This Assurance benchmark derives from retained row C-009 (Hadwiger–Nelson).
It tests exact-vs-numerical geometric reasoning: one wrong square-root branch
silently invalidates part of a named unit-distance graph.

The single primary objective is exact radical-branch diagnosis and repair. The
verifier reduces all 42 corrupted/corrected squared-distance claims in
`Q(sqrt(33))`, identifies false claimed edges, and reconstructs the corrected
unit graph. Provisional difficulty is Hard because the agent must preserve
nested radical signs across a complete pair audit; weaker systems commonly
square away the sign or use decimal tolerance.

The shortcut audit rejects label-only repair claims, partial edge checks,
floating-point evidence, and the public standard edge list without a complete
corrupted-state diagnosis. Full reward is `CHECKED` for this embedding only.

Source: SageMath's pinned `MoserSpindle` exact embedding and the retained
conjecture inventory row C-009. The authored corruption is not a claim about
the source implementation.
