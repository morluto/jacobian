# Audit the claimed uniqueness of an APMO path construction

The frozen ProofBench model response concludes that the only possible parameters are `alpha=beta=1`. Audit that conclusion.

Submit a nontrivial positive rational pair with `alpha >= beta`, `alpha+beta=2`, and `alpha != beta`. Equivalent rational spellings are accepted. Rational numerators and denominators use ordinary signed base-10 notation with at most 64 digits each; exponent notation is not accepted. Use the path `x_n=ceil(n/2)`, `y_n=floor(n/2)`. Provide an exact parity certificate showing that `floor(x_n alpha+y_n beta)=n` for every nonnegative integer `n`, and a 16-row exact trace.

The digest-bound `evidence/answer.txt` must contain exactly six nonempty lines: `emerald-path-family-certificate-v1`, the four fields `alpha:`, `beta:`, `even_offset:`, and `odd_offset:` using the submitted strings, then `trace_sha256:` followed by the SHA-256 of the submitted trace serialized as sorted-key compact JSON.

Include this published limitation exactly: `The certificate refutes the published singleton claim and proves sufficiency for its submitted family member; it does not independently prove necessity for every possible trip.` Do not claim `VERIFIED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier reports mathematical, evidence, input-binding, scope, completeness, limitations, protocol, and assurance dimensions separately. Rational certificate fields are bounded to at most 64 base-10 digits per numerator and denominator so parsing remains deterministic and resource-bounded.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PUBLISHED_SINGLETON_CLASSIFICATION_IS_FALSE`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
