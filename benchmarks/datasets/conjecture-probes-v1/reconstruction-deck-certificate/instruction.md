# Reconstruct a graph from its scrambled vertex deck

The input contains nine independently relabeled eight-vertex cards. Recover a
simple graph on vertices `0..8` with exactly 15 edges. For every card submit
the deleted original vertex and a bijective `local_to_original` map of its
local vertices `0..7` onto the other eight original vertices.

The verifier checks every mapped card edge set against the corresponding
vertex-deleted subgraph. It also aggregates all mapped cards and requires every
original edge to occur exactly seven times and every nonedge zero times.
Relabeled isomorphic reconstructions are accepted.

Evidence is matching JSON (`application/json`) with exactly `schema_version`
(the string `"1"`), `task_id`, `result`, and `limitations`. This reconstructs one finite deck only and
does not prove the Reconstruction Conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
