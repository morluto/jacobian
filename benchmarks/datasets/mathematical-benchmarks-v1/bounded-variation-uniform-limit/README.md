# Bounded-variation uniform-limit separation

This Regression benchmark transforms COUNTERMATH row 600 (`359.txt`, real
analysis) from `Sheaa/countermath_eval` at immutable revision
`d4e9f8ca877552f4491a9c2d52e0d230c0fca620` (CC-BY-SA-4.0).

The agent constructs a uniformly vanishing sequence of bounded-variation
functions whose total variations stay equal to four. The verifier independently
replays the general sup-norm and monotone-segment formulas and exact checkpoints
for a freely chosen integer frequency scale.

Family: **Regression**. Primary objective: **separate uniform convergence from
convergence of total variation**. Difficulty: **Hard (provisional)** because the
certificate combines a functional counterexample, a quantified uniform bound,
and exact variation accounting; baseline calibration may lower it to
Medium-Hard. The ceiling is `COMPUTED`: elementary trigonometric monotonicity is
trusted and no proof assistant checks the infinite statements.

Shortcut audit: the conclusion label, a graph, or finitely sampled function
values cannot pass. Alternative scale factors and checkpoint indices are
accepted. The workflow is distinct from path-dependent multivariable limits and
existing sequence convergence-mode tasks.
