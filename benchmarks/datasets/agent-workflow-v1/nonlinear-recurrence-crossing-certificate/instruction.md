# Certify a negative recurrence term

For the total real sequence `a₁=56`, `aₙ₊₁=aₙ-1/aₙ`, prove that some
`aₙ<0` with `n<2002`. Submit an exact certificate based on `dₙ=aₙ²`: give the
potential identity coefficients in the Laurent basis `[a², 1, a⁻²]`, a
rational threshold phase, its exact minimal step budget, and three exact
open-interval images under `a -> a-1/a` that force a negative term. The three
interval certificates may be listed in any order and are checked from their
rational endpoints rather than from prose labels. Use reduced rational
numerator/denominator objects; numerical simulation is not accepted.

Write `submission.json` and digest-bind
`evidence/nonlinear-recurrence-certificate.json`, which must copy `result` and
`limitations` exactly. Claim at most `COMPUTED`.
