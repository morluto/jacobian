# Repair an omitted Apollonius-circle computation

An OPC proof received human score zero after asserting, without computation, that the circle with diameter the internal/external ratio points implies `MA^2 = k^2 MB^2`.

Choose canonical positive rational `k != 1` and `c`. Put `A=(0,0)`, `B=(c,0)`, `P=kc/(k+1)`, and `Q=kc/(k-1)`. Submit `P,Q`, the center and positive radius of the diameter circle, and coefficient vectors in basis `[x^2,y^2,x,1]` for its expanded equation and for `MA^2-k^2 MB^2`. Submit the exact proportionality multiplier proving the omitted identity.

Every rational field uses canonical reduced syntax: an integer such as `12` or
a fraction such as `-3/5`, with no leading zeros, exponent notation, or `/1`.

The digest-bound `evidence/answer.txt` must contain exactly four nonempty lines: `apollonius-coefficient-certificate-v1`, `multiplier: <submitted multiplier>`, `circle_coefficients: <four submitted values joined by commas>`, and `distance_coefficients: <four submitted values joined by commas>`.

Include this published limitation exactly: `The certificate repairs the annotated coordinate identity; it does not independently formalize every theorem or endpoint convention in the full geometry proof.` Assurance ceiling is `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier reports mathematical, evidence, input-binding, scope, completeness, limitations, protocol, and assurance dimensions separately. Every rational field must use canonical reduced integer or numerator/denominator syntax with at most six digits per integer component; exponent notation, leading zeros, and denominator 1 are rejected.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `OMITTED_APOLLONIUS_IDENTITY_REPAIRED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
