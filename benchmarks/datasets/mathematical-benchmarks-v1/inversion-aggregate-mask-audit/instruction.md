# Audit a locally wrong inversion definition masked by a correct aggregate

The frozen formalization calls a pair `(i,j)` an inversion when `sigma[i] <= sigma[j]`; the intended definition uses `sigma[i] > sigma[j]`. For `n=4`, produce `/app/submission.json` following `/app/submission_schema.json` and `/app/evidence/inversion-audit.json` following `/app/evidence_schema.json`.

Supply any permutation witnessing different pointwise counts. Also report the independently computed sums of both counts over all 24 permutations. The witness is not fixed; valid alternatives are accepted. The verifier exhaustively recomputes both functions and both aggregates.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/inversion-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
