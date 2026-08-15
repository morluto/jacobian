# Binary fractional-ratio proof repair

This Regression benchmark freezes a public answer that silently replaces a binary ratio objective by a budget-constrained fractional-knapsack program. The primary objective is proof diagnosis and exact repair.

The verifier requires all three contract mismatches, then independently replays a 24-item residual optimality certificate. For a submitted ratio `p/q`, it checks `q t_i - p f_i` for every item, the affine constant residual, the selected set, the zero maximum residual, and the attained objective. This establishes the frozen optimum without trusting the public proof or exhaustively enumerating `2^24` vectors.

- **Family:** Regression.
- **Quality:** 86/100.
- **Difficulty:** Hard (provisional), due to multi-stage proof diagnosis and exact primal/dual-style bookkeeping.
- **Shortcut audit:** the instance has 24 items and requires the full residual certificate; copying the public fractional-knapsack route fails because no budget or fractional variables exist.
- **Portfolio value:** adds objective/feasible-set substitution repair in discrete fractional programming.

Source: `Jiahao004/DeepTheorem`, immutable revision `f5935720f176cedff4ecd8ebf83d1696e31cfac8`, train row 10007 / source id 86829, MIT. The verifier proves only the frozen exact optimization certificate; it does not machine-prove a general greedy theorem.
