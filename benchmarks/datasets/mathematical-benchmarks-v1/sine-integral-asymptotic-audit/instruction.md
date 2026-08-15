# Audit a published sine-integral asymptotic expansion

The frozen dataset answer claims

`Si(x) = PI/2 - cos(x)/x + sin(x)/x^2 + ...`.

Audit its signs without numerical sampling. Submit an exact certificate obtained from the tail
`J(x) = integral_x^infinity sin(t)/t dt`. Give five terms of `J`, the corresponding five terms of `Si = PI/2 - J`, and the single integral remainder after five integrations by parts. Your certificate must be independently checkable by differentiating the proposed tail identity.

Also give a rigorous absolute bound for the scaled remainder, using
`|integral_x^infinity cos(t)/t^6 dt| <= 1/(5*x^5)` for `x > 0`, and classify whether the published coefficient of `sin(x)/x^2` is correct.

Write `/app/submission.json` and exactly one task-specific witness file at `/app/evidence/answer.txt`. The evidence must contain exactly four nonempty lines: `sine-integral-certificate-v1`, `result_sha256: <digest>` where the digest is SHA-256 of the submitted result serialized as sorted-key compact JSON, `published_sine_coefficient: <submitted integer>`, and `corrected_sine_coefficient: <submitted integer>`.


<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
