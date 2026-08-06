# Discover and certify a reciprocal-polynomial solution

Find a nonconstant integer-coefficient polynomial `P` of degree between 11 and
39 satisfying `1/P(z)+1/P(1/z)=z+1/z`. Discover the relevant family index `m`
between 6 and 20, then submit an exact sparse coefficient list for `P`, the
coefficients of `Q=P/z`, the reversed polynomial `S=z^deg(Q) Q(1/z)`, and the
constant relating `S` and `Q`. The verifier will independently recognize the
family member, check the geometric divisibility identity, and reconstruct the
cleared Laurent identity. Numerical sampling is not accepted.

Write `submission.json` and digest-bind
`evidence/classification-certificate.json`, which must copy `result` and
`limitations` exactly. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `FAMILY_MEMBER_CERTIFIED`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/classification-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/classification-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
