# Series-domain junk-zero assurance audit

This Assurance benchmark is derived from formal-conjectures issue #3931 and source commit `60d13721bee9c49a21ee6510e2cf924637e47725`. It isolates a trust-boundary failure: a Dirichlet-series implementation returns zero when its defining series is not summable, creating zeros outside the analytic domain.

The agent chooses a reciprocal exponent, supplies the general affine exponent of the dyadic lower bound, and replays nine exact instances. The verifier recomputes that symbolic relation and every frozen exponent, then confirms that the zero follows from the fallback convention rather than analytic continuation. The finite window is construction-replay evidence; the divergence conclusion uses the submitted general bound. Difficulty is **Hard (provisional)**. The result is a frozen API-domain audit, not a theorem about the genuine Dedekind zeta continuation.
