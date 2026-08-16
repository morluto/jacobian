# Exact polynomial-map claim assessment

Assess the `bounded-collision-scope` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are inputs to audit, not authority. Return the
terminal certificate in the `result` described by `submission_schema.json`.
Its bindings must identify the exact claim, map, subject, semantics, and
checker identities frozen in the input.

This bounded-search claim licenses only a `COLLISION_WITNESS_REPLAY` certificate. Any mathematically valid collision witness in the declared grid is acceptable. The terminal `verdict` is `COLLISION_FOUND`. That `verdict` is a semantic field of the
mathematical result; do not add a separate generic conclusion or assurance
claim.

Use any mathematical method. No external service or special tool is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit the family-licensed terminal certificate inside result. The verifier replays the exact polynomial-map predicate and checks the frozen claim bindings carried by the result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
