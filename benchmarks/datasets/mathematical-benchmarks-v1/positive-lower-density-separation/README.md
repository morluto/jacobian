# Positive-lower-density semantic separation

This Assurance benchmark is derived from formal-conjectures issue 4553, which reports that an Erdős problem's intended “positive lower density” was strengthened to “the density exists and is positive.” The agent must construct a parameterized alternating geometric-block set whose lower density is positive but whose natural density does not exist.

The verifier independently recomputes eight exact endpoint counts and density ratios for any base from 2 through 9, checks the closed-form count and two distinct subsequential limits, and rejects finite-window or `VERIFIED` overclaims.

- **Quality:** 87/100.
- **Difficulty:** Hard (provisional): parameterized construction, exact endpoint arithmetic, and semantic interpretation are all required.
- **Shortcut audit:** no fixed base is required; eight levels and exact formulas prevent label-only answers. The verifier accepts alternative bases.
- **Portfolio value:** density-existence strengthening is distinct from quantifier polarity and convergence-mode implication audits.

The finite replay checks instances of the submitted general formula; it is not itself a machine proof over all natural indices or real limits. Assurance is capped at `COMPUTED`.
