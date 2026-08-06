# Construct a fiber bijection by unique path lifting

The frozen input gives a finite graph covering and two base vertices. Choose
any simple base path satisfying the declared length bounds. Lift that path from
every point in the source fiber, then lift the reversed path from every
resulting endpoint. Submit the complete forward and reverse lift traces and
the induced fiber bijection.

Your certificate must show that every trace projects to its declared base
path, every step is a cover edge, each lift is the unique available continuation,
and reverse lifting returns every source point. Merely reporting equal fiber
cardinalities is incomplete. Write the structured result to `submission.json`
using `submission_schema.json`, explain the construction in
`evidence/answer.txt`, and bind that evidence by SHA-256.

The verifier checks this finite covering exactly but does not certify the
general topological theorem, so claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `FIBERS_BIJECTED_BY_PATH_LIFTING`, `INVALID_COVER`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
