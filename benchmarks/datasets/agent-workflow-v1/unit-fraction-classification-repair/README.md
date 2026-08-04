# Unit-fraction classification repair

Regression benchmark derived from Xerv-AI/GRAD train row 27 at immutable revision `71595210590450202b7b69225bc07e9e01b13c5c` (MIT). It repairs a long but invalid Diophantine classification and binds the corrected finite result to complete coverage evidence.

Family: Regression. Primary objective: proof repair. Quality score: 88/100. Difficulty: Hard (provisional): success requires isolating invalid divisibility inferences, deriving the exact divisor interval, and producing a complete 2025-case certificate rather than trusting the published answer.

Shortcut audit: the published count 2022, a few examples, or an unbound list cannot pass. The verifier independently enumerates every divisor interval, compares the bit-packed membership vector, reconstructs `(x,y)` for varied witnesses, and checks counterexamples. Scope is bounded and assurance remains `COMPUTED`.
