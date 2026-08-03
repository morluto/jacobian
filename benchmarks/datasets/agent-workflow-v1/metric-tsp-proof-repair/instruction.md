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
