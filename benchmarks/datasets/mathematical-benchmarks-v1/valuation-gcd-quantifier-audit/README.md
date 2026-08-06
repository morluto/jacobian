# Valuation gcd quantifier audit

Hard-provisional Regression benchmark derived from Xerv-AI/GRAD train row 83.
The source proof translates `gcd(a,b,c,d)=1` as requiring a zero minimum
valuation for at least one prime. The correct condition quantifies over every
prime dividing the lcm.

The task requires a countermodel to the weakened condition and a separate
valuation table satisfying the repaired gcd condition together with the local
lcm and product constraints. The verifier recomputes all prime-power
consequences; it does not adjudicate the source problem's final summation.

Quality score: **86/100**. The difficulty is provisional pending model
calibration. Knowing the corrected quantifier is insufficient without two
independently checked, freely chosen valuation systems.
