# Certify a cubic root bound symbolically

For real parameters `1 >= a >= b >= c >= 0`, prove that every complex root of
`P(x)=x^3+a*x^2+b*x+c` has modulus at most one.

Submit the reciprocal-polynomial reduction and a coefficient-difference
certificate for `Q(z)=1+a*z+b*z^2+c*z^3`. Represent every affine expression by
its exact integer coefficient vector in basis `[1,a,b,c]`. Your certificate
must expose four nonnegative weights, their telescoping sum, the coefficients
of `(1-z)Q(z)`, the rearranged root identity, and the powers controlled when
`|z|<1`. Do not use numerical sampling.

Write `submission.json` to the supplied schema. Write
`evidence/root-bound-certificate.json` with exactly `schema_version`, `task_id`,
`result`, and `limitations`, copy the corresponding submission values exactly,
and bind it by SHA-256. Claim at most `COMPUTED` and use limitation code
`ELEMENTARY_COMPLEX_MODULUS_LEMMAS_TRUSTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/root-bound-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
