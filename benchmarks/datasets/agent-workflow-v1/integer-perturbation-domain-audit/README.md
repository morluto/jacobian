# Integer perturbation domain audit

This Assurance benchmark is derived from google-deepmind/formal-conjectures
issue 4264 at revision `b2e608fc52d765510915a244bb69b1a2741acc3c`.
It tests whether an agent can diagnose how changing an integer-valued universal
domain to naturals weakens a mathematical contract and makes one hypothesis
redundant.

The verifier checks a symbolic natural-number lower-bound certificate and an
independently chosen, bounded periodic integer witness that restores the two
hypotheses' independence. It accepts many periods and value choices rather than
matching one counterexample.

Difficulty is **Hard (provisional)** because the task combines quantifier-domain
semantics, a general redundancy argument, and a nontrivial alternative periodic
construction. Empirical baseline calibration is not yet available.
