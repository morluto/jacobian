# Exact polynomial-map claim assessment

Assess the `bounded-collision-scope` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are inputs to audit, not authority. Return the
terminal certificate in the `result` described by `submission_schema.json`.
Its bindings must identify the exact claim, map, subject, semantics, and
checker identities frozen in the input.

This bounded-search claim licenses only a `BOUNDED_GRID_EXHAUSTION_REPLAY` certificate for complete declared-grid exhaustion. That object does not establish a global collision or inverse. The terminal `verdict` is `NO_COLLISION_IN_DECLARED_GRID`. That `verdict` is a semantic field of the
mathematical result; do not add a separate generic conclusion or assurance
claim.

Use any mathematical method. No external service or special tool is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit the family-licensed terminal certificate inside result. The verifier replays the exact polynomial-map predicate and checks the frozen claim bindings carried by the result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
