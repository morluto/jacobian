# Solve a functional equation through its Möbius orbit

Find every function `F : R \ {0,1} -> R` satisfying

`F(x) + F((x-1)/x) = 1+x`.

Submit the three rational functions in the orbit of `x` under
`T(x)=(x-1)/x`, the corresponding right-hand sides, the three rational
functions giving `F` on that orbit, and the exact integer coefficient matrix
whose nonsingularity establishes uniqueness. Rational functions are encoded
as primitive integer coefficient arrays in ascending degree order, with a
positive leading denominator coefficient. The verifier independently replays
the Möbius cycle and every rational-function equation. A copied final formula
without the orbit certificate cannot pass.

Write `submission.json` and digest-bind
`evidence/functional-equation-certificate.json`, which must copy `result`
and `limitations` exactly. Claim at most `COMPUTED`.
