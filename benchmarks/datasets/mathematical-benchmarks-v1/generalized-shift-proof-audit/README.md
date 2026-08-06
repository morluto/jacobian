# Generalized-shift proof audit

This Research Diagnostic benchmark freezes GRAD train row 26 and asks for four
exact certificates showing that its generated proof is invalid. It adds a
multi-defect operator-theory audit rather than another answer-recovery task.

The verifier checks a noninjective bounded-displacement shift, the scaled DFT
norm from orthogonality, an exact diagonal norm counterexample, and the real
domain of a radical. It accepts alternative indices, Fourier sizes, diagonal
entries, and integers. Its assurance ceiling is `COMPUTED`; no conclusion about
the original supremum is made.

Difficulty is provisionally Hard because success requires separating four
different mathematical failure modes and producing compatible exact evidence.
The shortcut audit found no tiny published witness that completes all four
obligations, and simple answer matching cannot satisfy the verifier.
