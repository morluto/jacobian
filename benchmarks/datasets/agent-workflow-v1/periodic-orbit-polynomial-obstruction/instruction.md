# Certify the periodic-orbit polynomial obstruction

Let `F(n)` be the finite number of fixed points of `T^n` for a function on the
integers. Prove that `F(n)=P(n)` for every positive integer `n` is impossible
when `P` is a nonconstant integer polynomial.

Submit a structured certificate using two distinct symbolic primes `p,q`. It
must expose the exact-period Möbius coefficient vector at `pq`, its divisibility
by `pq`, both reductions modulo `p` and modulo `q`, and the final infinite-prime
and polynomial-identity steps. Use basis `[F(pq),F(p),F(q),F(1)]` for the orbit
coefficient vector and `[P(q),P(1)]` or `[P(p),P(1)]` for modular residues.

Write `submission.json` to the supplied schema. Write
`evidence/periodic-orbit-certificate.json` with exactly `schema_version`,
`task_id`, `result`, and `limitations`, copying the corresponding submission
values exactly and binding the file by SHA-256. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `NONCONSTANT_POLYNOMIAL_IMPOSSIBLE`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/periodic-orbit-certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/periodic-orbit-certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
