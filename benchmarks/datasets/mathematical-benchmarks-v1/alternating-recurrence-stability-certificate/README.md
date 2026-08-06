# Alternating recurrence stability certificate
Regression benchmark from ByteDance-Seed/BeyondAIME test row 2 at immutable revision `c705198ae1043810b1e1693bd879250b51a7a523` (CC0-1.0), canonical row digest `sha256:27dc6f3fbe550296209986c9e0d23532b67ed064e507167d80a6a3f66b9e0d0c`.

The task asks for the exact particular/homogeneous decomposition of an alternating recurrence, a parity-sensitive instability argument for every nonzero perturbation, and exact monotonicity checkpoints for the unique stable initial value.

Family: Regression. Primary objective: recurrence stability reasoning. Quality score: 86/100. Difficulty: Hard (provisional); the infinite parity/dominance step is the discriminator, while empirical calibration remains pending.

Shortcut audit: the published answer `9`, a finite simulation, or the fixed-point coefficient alone cannot pass. The verifier accepts varied checkpoint indices, independently replays the recurrence and difference formulas, and requires both perturbation-sign branches plus the exponential dominance ratio. The Archimedean dominance inference is trusted, so assurance is `COMPUTED`.
