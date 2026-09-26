# Cubic resolvent of a monic quartic

For a monic quartic

\[f(T)=T^4+aT^3+bT^2+cT+d\]

with roots \(\alpha_1,\ldots,\alpha_4\), this operation returns the cubic
whose roots are the three pair-product sums

\[\alpha_1\alpha_2+\alpha_3\alpha_4,\quad
  \alpha_1\alpha_3+\alpha_2\alpha_4,\quad
  \alpha_1\alpha_4+\alpha_2\alpha_3.\]

This fixes the convention explicitly; the returned polynomial in \(Y\) is

\[Y^3-bY^2+(ac-4d)Y+(4bd-a^2d-c^2).\]

The result carries the source quartic and the cubic as monic univariate
polynomials over `QQ`. The request admits coefficients with numerator and
denominator components of at most 306 decimal digits, derived from the
operation's bounded exact-work and output budgets. Requests beyond those
budgets are refused as resource admission failures before coefficient
arithmetic.

As an independent fixture, roots \(1,2,3,4\) give pair-product sums
\(14,11,10\), hence \(Y^3-35Y^2+404Y-1540\).
