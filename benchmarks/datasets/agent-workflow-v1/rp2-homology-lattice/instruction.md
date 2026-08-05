# Integral homology certificate for a triangulated projective plane

The frozen input gives ten triangular facets on vertices `0..5`.  Using the
increasing orientation for every edge and triangle, produce a replayable
integer-chain certificate that the first homology group is `Z/2Z`.

Choose any spanning tree of the one-skeleton.  Order the ten non-tree edges and
the ten facets.  For each facet boundary, give its coordinates in the
fundamental-cycle basis determined by the tree: the matrix entry at row `i`,
column `j` is the coefficient of non-tree edge `i` in the boundary of facet `j`
(with the increasing-orientation boundary convention).  Either the
edge-per-row or facet-per-row orientation is accepted — the verifier checks
both against the canonical boundary matrix.  Report the resulting square
integer matrix, its determinant, and the homology conclusion.

The verifier reconstructs the one-skeleton, checks the tree, boundary-of-
boundary identities, independently derives every cycle coordinate, and
computes the determinant exactly.  Alternative spanning trees and orderings
are accepted.  Submit `/app/submission.json` following the supplied schema and
one bound explanation at `/app/evidence/answer.txt` that states the cycle
lattice calculation, determinant `-2`, lattice index `2`, and `Z/2Z`
conclusion.

Do not claim proof-assistant verification.  This task provides exact finite
chain-complex computation only.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `INTEGRAL_H1_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
