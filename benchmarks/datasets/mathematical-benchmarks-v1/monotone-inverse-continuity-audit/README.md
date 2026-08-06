# Monotone inverse continuity audit

This Assurance benchmark is derived from TaoAnalysisBench
`taobench_000453_textbook` at immutable revision
`339937d75342072a31903739b1bbbe72e1b40c21` (CC-BY-4.0). It isolates the
continuity hypothesis in the inverse theorem by requiring a parameterized,
exactly checked jump countermodel rather than a yes/no label.

## Selection and portfolio value

The case adds missing-hypothesis diagnosis for interval images and inverses.
Nearby elementary series and finite-sum rows were rejected as routine or
already represented. The task accepts a family of rational countermodels and
therefore does not reduce to recalling one public witness.

## Family, objective, and difficulty

- Family: Assurance.
- Primary objective: diagnose the semantic effect of omitting continuity.
- Difficulty: Hard (provisional), based on the multi-stage obligation to
  construct a globally strictly increasing discontinuous map, compute its
  exact image gap, and connect that gap to failure of the inverse contract.

Weaker models are expected to confuse strict monotonicity with continuity or
check only the two branches separately. Stronger models should bind all image
values and the omitted point. Tool-less agents can solve it, but cannot rely on
sampling alone because the exact global branch inequalities are required.

## Shortcut and assurance audit

The verifier accepts alternative bounded rational parameters and gap witnesses,
recomputes all values, checks both branch slopes and the cross-branch jump, and
rejects duplicate evidence, malformed rationals, incorrect scope, and false
`VERIFIED` claims. It establishes the declared piecewise-family countermodel
only; it does not machine-check the original Lean declaration or arbitrary real
functions. The ceiling is therefore `COMPUTED`.
