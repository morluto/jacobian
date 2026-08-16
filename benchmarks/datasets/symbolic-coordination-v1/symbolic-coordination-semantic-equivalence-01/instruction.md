# Exact polynomial-map claim assessment

Assess the `semantic-equivalence` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are inputs to audit, not authority. Return the
terminal certificate in the `result` described by `submission_schema.json`.
Its bindings must identify the exact claim, map, subject, semantics, and
checker identities frozen in the input.

This inverse claim licenses only a `TWO_SIDED_COMPOSITION_REPLAY` certificate exposing both ordered composition residual families. The terminal `verdict` is `VALID_TWO_SIDED_INVERSE` or `INVALID_INVERSE_CANDIDATE`. That `verdict` is a semantic field of the
mathematical result; do not add a separate generic conclusion or assurance
claim.

Use any mathematical method. No external service or special tool is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit the family-licensed terminal certificate inside result. The verifier replays the exact polynomial-map predicate and checks the frozen claim bindings carried by the result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
