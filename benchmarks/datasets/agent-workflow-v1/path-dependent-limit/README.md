# Path-dependent limit certificate

This Regression benchmark transforms COUNTERMATH row 675 (`426.txt`, real
analysis) from the public `Sheaa/countermath_eval` mirror at immutable revision
`d4e9f8ca877552f4491a9c2d52e0d230c0fca620` (CC-BY-SA-4.0).

The agent constructs a member of the family
`x^(2p)y/(x^(4p)+y^2)`.  A clean-room verifier checks the symbolic order
calculation for every straight line, the two coordinate axes, and freely chosen
nonlinear rational paths `y=c*x^(2p)` that give nonzero limiting values.  It
accepts several exponents and path parameters rather than matching one public
witness.

Family: **Regression**. Primary objective: **construct and certify a
multivariable path-dependent limit**. Difficulty: **Hard (provisional)** because
the response must coordinate a parameterized construction, a universal
straight-line order argument, and exact nonlinear-path witnesses; empirical
calibration may place it at Medium-Hard. The assurance ceiling is `COMPUTED`:
the verifier replays exact algebraic certificates but does not formalize real
limits in a proof assistant.

Shortcut audit: a remembered example, decimal sampling, or one nonlinear path
without the all-lines order certificate fails. Existing tasks do not test the
distinction between all linear approaches and the full multivariable limit.
