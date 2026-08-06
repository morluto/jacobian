# Newton-polygon factorization audit

Hard-provisional Regression benchmark derived from
`LukeBailey181Pub/ArxivMathGradingBench` train row 32 at revision
`86c2b07ec545c0bd37feac10d4fc03675a85a6f6` and the CC BY 4.0 source versions
of arXiv:2505.08549. The dataset-row SHA-256 is
`9e6998b220ebd9cc4fa966326e216e2b69f0a7de9a8c3174b8edaf07abf60c2d`;
the v1/v2 source archive SHA-256 values are
`76057945d543d1fabc79971ce51019cd59584ed60021f91565c45011a7570e2d` and
`941dc5c03d46dfb8a06ce02ea4a04877d301acc0cfde0a92bdee69afe8c63aab`.

The first paper version inferred a constant-term valuation conclusion from a
single primitive rightmost Newton-polygon edge. The correction adds control of
the left edge and changes the conclusion. This task asks for a non-tiny exact
factorization that satisfies the old hypotheses but refutes its conclusion and
fails at least one added left-edge hypothesis.

The verifier independently multiplies the integer factors, computes exact
`p`-adic valuations, constructs the lower Newton hull, checks the old slope and
primitivity conditions, rejects the old conclusion, and checks why the repaired
hypotheses do not apply. It accepts any qualifying prime and factorization.

Family: **Regression**. Primary objective: **diagnose a literature proof scope
error by constructing a replayable Newton-polygon counterexample**. Quality
score: 89/100. Difficulty is Hard (provisional): the task requires coordinated
factor construction, convolution, valuation inequalities, hull geometry, and
repair-boundary analysis; baseline calibration is pending.

Shortcut audit: no counterexample is supplied by the dataset row or either
paper version. Both factors must have degree at least two and the product degree
at least six, preventing the tiny linear example from dominating. The verifier
accepts alternative witnesses and does not compare against the Oracle values.

Assurance is capped at `COMPUTED`: exact finite polynomial and valuation facts
are replayed, but Dumas's theorem and the corrected general lemma are not
machine-formalized.
