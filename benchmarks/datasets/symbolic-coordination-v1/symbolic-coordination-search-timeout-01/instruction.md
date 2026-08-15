# Exact polynomial-map claim assessment

Assess the `bounded-collision-scope` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are evidence to audit, not authority. Return the
terminal certificate described by `submission_schema.json`, bind it to the
exact claim, map, subject, semantics, and checker identities in the input, and
write the mirrored JSON certificate to
`evidence/certificate.json` with its SHA-256 digest.

For an inverse claim, the certificate must expose both ordered composition
residual families. A Keller-condition certificate licenses only its exact
constant nonzero Jacobian claim. A bounded search licenses only an exact
collision witness, complete declared-grid exhaustion, or an honest
non-conclusion matching timeout or incomplete execution. Any mathematically
valid collision witness in the declared grid is acceptable. The terminal
`verdict` is a semantic field of the mathematical result; do not add a
separate generic conclusion or assurance claim.

Use any mathematical method. No external service or special tool is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Write evidence/certificate.json as a JSON wrapper with exactly these fields: schema_version (the string \"1\"), task_id (the string \"jacobian/symbolic-coordination-search-timeout-01\"), result (an exact copy of the submission result object). Bind that exact regular file by SHA-256. The verifier independently replays the mathematics and checks the input, artifact, and witness identities.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/certificate.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
