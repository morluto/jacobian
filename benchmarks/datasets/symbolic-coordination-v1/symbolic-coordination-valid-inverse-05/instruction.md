# Exact polynomial-map claim assessment

Assess the `valid-two-sided-inverse` claim frozen in `input.json` under exact rational
polynomial semantics. Supplied candidates, provider statuses, partial direction
checks, and search records are inputs to audit, not authority. Return the
terminal certificate in the `result` described by `submission_schema.json`.
Its bindings must identify the exact claim, map, subject, semantics, and
checker identities frozen in the input.

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

Submit the terminal certificate inside result. The verifier replays the exact polynomial-map predicate and checks the frozen claim bindings carried by the result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
