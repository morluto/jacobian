# Lp integrability separator

This Regression benchmark freezes CounterMATH test row 611 at revision
`d4e9f8ca877552f4491a9c2d52e0d230c0fca620`. The agent must construct a
nonnegative measurable function whose only positive finite Lebesgue exponent is
`p = 2`, using a two-tail power/logarithm family. The exponent `beta` is not
fixed: every canonical rational `beta > 1/2` is accepted.

The clean-room verifier recomputes the transformed `p=2` integrals and the
power thresholds at zero and infinity. It relies only on the standard
power-log integral criterion; it does not numerically sample the function and
does not claim proof-assistant verification. The assurance ceiling is
`COMPUTED`.

## Curation and difficulty

The source was selected because it requires one construction to balance two
opposing singular regimes. Nearby one-path limit and elementary discontinuity
counterexamples were rejected as shorter workflows already represented in the
portfolio. A public answer alone is insufficient because the verifier accepts
an infinite family of rational parameters and reconstructs every threshold.

Difficulty is **Hard (provisional)**: weaker agents often repair one tail while
breaking the other, or overlook the logarithmic boundary at `p=2`; stronger
agents should derive the two substitutions and calibrate the assurance. A
tool-less agent can solve it, but not by tiny finite search or answer-pattern
matching.

## Shortcut audit

The task disallows a fixed witness lookup, numeric quadrature, and finite probe
claims. The submitted parameter, two exact transformed integrals, and both
universal power-regime certificates must agree. Memorizing one common example
does not bypass the independent recomputation.
