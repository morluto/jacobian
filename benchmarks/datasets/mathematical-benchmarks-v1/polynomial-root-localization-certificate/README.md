# jacobian/polynomial-root-localization-certificate

Construct a symbolic coefficient-difference certificate proving that every
root of `x^3 + a*x^2 + b*x + c`, under `1 >= a >= b >= c >= 0`, lies in the
closed unit disk.

This Regression benchmark is derived from `lm-provers/FineProofs-SFT` train row
330 at immutable revision `73661e62811cf2940a0d3f82788a4f4332204c2f`
(Apache-2.0). The verifier does not trust the public proof: it checks exact
affine coefficient vectors for the four nonnegative differences, their
telescoping sum, multiplication of the reciprocal polynomial by `1-z`, and the
resulting modulus-domination contradiction.

## Quality and shortcut audit

Quality score: **86/100**. The single primary objective is symbolic root-bound
certification. It adds reciprocal-polynomial localization and a general
coefficient-difference argument, distinct from existing factorization,
normalization, and finite root checks. Difficulty is **Hard (provisional)**:
the proof requires a reciprocal reduction, a non-obvious telescoping transform,
and an inequality chain, but empirical calibration is unavailable.

The task cannot be passed by reporting `|lambda| <= 1` or by checking sample
coefficients. The complete symbolic identity must hold in the basis
`[1,a,b,c]`. The verifier treats the triangle inequality and `r^k <= r` for
`0 <= r < 1` as explicit trusted elementary lemmas; it does not certify complex
analysis beyond that frozen argument and therefore caps assurance at `COMPUTED`.
