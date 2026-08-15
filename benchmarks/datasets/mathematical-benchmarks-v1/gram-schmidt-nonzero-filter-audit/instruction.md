# Gram-Schmidt filter semantic audit

Audit whether the frozen formal filter preserves the informal promise to remove
vectors that become zero during Gram-Schmidt.

Construct six distinct nonzero integer vectors in `Q^5`. The first four must
have all five coordinates nonzero and be linearly independent. The last two
must each have at least two nonzero coordinates, and the whole sequence must
have rank exactly four. Submit every exact unnormalised Gram-Schmidt residual
as reduced rational coordinates, together with the zero-residual indices and
the two filter outcomes. The verifier independently replays exact rational
Gram-Schmidt and rank computation; valid alternative vector systems pass.

Write one digest-bound evidence object at
`/app/evidence/gram-schmidt-audit.json`, containing exactly `schema_version`,
`task_id`, `result`, and `limitations`. Only `COMPUTED` is supported. The task
does not elaborate Lean or machine-check the surrounding Mathlib theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact rational finite-dimensional replay; Lean elaboration remains outside scope.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/gram-schmidt-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
