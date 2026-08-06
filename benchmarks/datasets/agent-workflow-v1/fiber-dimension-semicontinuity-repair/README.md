# Fiber-dimension Fitting-ideal certificate

Hard-provisional Regression benchmark derived from DeepTheorem train row 10018
at revision `f5935720f176cedff4ecd8ebf83d1696e31cfac8`. The source row is MIT
licensed; its canonical row SHA-256 is
`8e8c71c00b23049df34c6a4de70f78d7fedecfede4c6a7547eb5c4c7a184ebc3`.

The task starts from two disclosed gaps in a generated proof of upper semicontinuity:
residue fields are not generally flat over their local rings, and locally closed
descriptions cannot be globalized by taking an arbitrary union. The agent must
repair the argument with right exactness and a global determinantal/Fitting
ideal, then certify the repair on a symbolic affine presentation. The verifier
computes all minors, checks equality of polynomial ideals in both directions by
Groebner reduction, and independently recomputes every frozen fiber rank.

- **Benchmark family:** Regression
- **Primary reasoning objective:** determinantal certificate construction from disclosed proof defects
- **Quality score:** 89/100
- **Difficulty:** Hard (provisional). The task combines module-presentation
  semantics, symbolic minors, ideal equality, and fiber-rank replay. The source
  proof itself is not agent-visible, so results must not be interpreted as
  measuring blind proof diagnosis. Empirical calibration has not yet been run.
- **Shortcut audit:** the source proof contains the flawed route but not the
  required symbolic repair certificate. Alternative ideal generating sets are
  accepted, so copying a fixed answer or exploiting generator order is not
  sufficient. The fixture is multivariate and excludes a tiny scalar witness.
- **Assurance boundary:** `COMPUTED`. The verifier checks the frozen affine
  presentation and the two identified proof obligations; it does not formalize
  the full scheme-theoretic theorem or certify the source proof as a whole.

Nearby DeepTheorem candidates were rejected when they reduced to routine
calculation, repeated an existing counterexample workflow, or required a large
unavailable geometric kernel. This row adds a new Fitting-ideal proof-repair
certificate workflow with an exact clean-room adjudicator.
