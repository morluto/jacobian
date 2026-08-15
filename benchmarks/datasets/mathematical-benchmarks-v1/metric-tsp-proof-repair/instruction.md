# Repair a Metric TSP proof

Audit and repair the flawed claim in the offline input. Identify the invalid
inference and state the strongest generic guarantee supported by the
double-tree argument. For the supplied metric, give a minimum spanning tree,
an Euler circuit of its doubled edges, the Hamiltonian cycle obtained by
retaining first visits, and an optimal Hamiltonian cycle. Report all exact
weights so the concrete trace demonstrates why exactness is unsupported while
the repaired guarantee holds.

Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a concise proof repair and weight calculation in `evidence/answer.txt`,
include a `RESULT_JSON:` line containing the submitted result as JSON, and bind
the file with its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
