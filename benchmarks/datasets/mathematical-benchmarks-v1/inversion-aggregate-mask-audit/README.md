# Inversion aggregate-mask audit

Assurance benchmark derived from AI-MO/CombiBench test row 20 at immutable revision `882ba08befd0856f5364db1e53d58c7e2cf704f9` (MIT). It tests detection of a local semantic defect that is invisible to the stated aggregate theorem.

Family: Assurance. Primary objective: semantic defect localization under aggregate masking. Quality score: 89/100. Difficulty: Hard (provisional): weaker systems often validate only the theorem's final total, while success requires separating pointwise and aggregate semantics and producing a replayable counterexample.

Shortcut audit: the public aggregate answer alone fails. The verifier accepts any discriminating permutation, exhaustively recomputes all 24 permutations, checks complementarity, and rejects false certification. The bounded `n=4` replay does not elaborate Lean and therefore remains `COMPUTED`.
