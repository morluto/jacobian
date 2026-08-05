# Exact polynomial-map claim assessment

Assess the `semantic-equivalence` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are evidence to audit, not authority. Return the
terminal certificate described by `submission_schema.json`, bind it to the
exact claim, map, subject, semantics, scope, and checker identities in the
input, and write the mirrored JSON certificate to
`evidence/certificate.json` with its SHA-256 digest.

For an inverse claim, the certificate must expose both ordered composition
residual families. A Keller-condition certificate licenses only its exact
constant nonzero Jacobian claim. A bounded search licenses only an exact
collision witness, complete declared-grid exhaustion, or an honest
non-conclusion matching timeout or incomplete execution. Any mathematically
valid collision witness in the declared grid is acceptable.

The `conclusion` field is determined by the terminal `verdict`:

| Verdict | Conclusion |
|---|---|
| `VALID_TWO_SIDED_INVERSE` | `TRUE` |
| `INVALID_INVERSE_CANDIDATE` | `FALSE` |
| `KELLER_CONDITION_ONLY` | `TRUE` |
| `NOT_KELLER` | `FALSE` |
| `COLLISION_FOUND` | `TRUE` |
| `NO_COLLISION_IN_DECLARED_GRID` | `FALSE` |
| `UNKNOWN` | `UNKNOWN` |

Use any mathematical method. No external service or special tool is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Write evidence/certificate.json as a JSON wrapper with exactly these fields: schema_version (the string \"1\"), task_id (the string \"jacobian/symbolic-coordination-semantic-equivalence-01\"), result (an exact copy of the submission result object), scope (an exact copy of the submission scope), completeness (an exact copy of the submission completeness), and limitations (an exact copy of the submission limitations). Bind that exact regular file by SHA-256. The verifier independently checks the mathematics, input and artifact identities, declared scope, completeness, and assurance.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `TRUE`, `FALSE`, `UNKNOWN`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/certificate.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
