# One-chart affine plane-curve blow-up

`algebraic_geometry.plane_curve.blowup_chart.compute` takes a nonzero bounded
affine curve polynomial over `QQ` and a rational point on the curve. For source
coordinates `(x,y)` and center `(a,b)`, it uses the chart
`x = a + u`, `y = b + u*t`. If the polynomial pulls back to `u^m G(u,t)`,
the operation returns the exact multiplicity `m`, strict transform `G`, and
the monic polynomial `G(0,t)` describing the strict transform's intersection
with the exceptional divisor in this chart. The source variables are retained
on the source polynomial and point; the two output variables are explicit,
fresh ordered axes.

This reports one affine chart. It does not claim the complementary chart's
intersection data or assemble global divisor classes. Expansion, degree,
term count, and coefficient growth are admitted before substitution; the
growth bound accumulates the distinct source coefficient denominators and the
center-coordinate powers up to the curve degree. Bounded exact QQ arithmetic
establishes the pullback divisibility identity.
