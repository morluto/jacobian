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

The verifier replays the task-specific mathematical predicate from the submitted result and validates a witness only where the task needs one.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
