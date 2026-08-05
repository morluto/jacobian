# Audit a published sine-integral asymptotic expansion

The frozen dataset answer claims

`Si(x) = PI/2 - cos(x)/x + sin(x)/x^2 + ...`.

Audit its signs without numerical sampling. Submit an exact certificate obtained from the tail
`J(x) = integral_x^infinity sin(t)/t dt`. Give five terms of `J`, the corresponding five terms of `Si = PI/2 - J`, and the single integral remainder after five integrations by parts. Your certificate must be independently checkable by differentiating the proposed tail identity.

Also give a rigorous absolute bound for the scaled remainder, using
`|integral_x^infinity cos(t)/t^6 dt| <= 1/(5*x^5)` for `x > 0`, and classify whether the published coefficient of `sin(x)/x^2` is correct.

Write `/app/submission.json` and exactly one evidence file at `/app/evidence/answer.txt`. The evidence must contain exactly four nonempty lines: `sine-integral-certificate-v1`, `result_sha256: <digest>` where the digest is SHA-256 of the submitted result serialized as sorted-key compact JSON, `published_sine_coefficient: <submitted integer>`, and `corrected_sine_coefficient: <submitted integer>`.

Include this published limitation exactly: `The checker replays an exact formal tail identity and bound under standard calculus lemmas; it does not machine-prove those lemmas or arbitrary transcendental asymptotics.` Do not claim `VERIFIED`; the assurance ceiling is `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier reports mathematical, evidence, input-binding, scope, and assurance dimensions separately.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PUBLISHED_SINE_TERM_SIGN_IS_WRONG`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
