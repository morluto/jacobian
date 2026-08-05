# Audit a noncompact Lefschetz argument

The frozen input contains a lemma used by two model answers: for a proper map of
a boundaryless manifold with finite-dimensional compact-support cohomology, no
fixed points supposedly imply zero compact-support Lefschetz number.

Audit that lemma and its use in the downstream torsion-freeness proof.

1. Give a nonzero rational translation of the real line.
2. Record the fixed-point equation and show it has no solution.
3. Record the compact-support cohomology dimensions, induced top-degree action,
   and exact compact-support Lefschetz number.
4. Identify the invalid compact-support step in the frozen graph/diagonal proof.
5. State precisely what the counterexample invalidates and what it does not
   decide about the original First Proof research question.

Write `submission.json` to the exact agent-visible schema. Put a concise audit
in `evidence/answer.txt`, include one `RESULT_JSON:` line containing the exact
submitted result as compact JSON, and bind that file with its SHA-256 digest.
Do not claim theorem verification; the verifier checks only this frozen
counterexample and argument boundary.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `FROZEN_LEMMA_REFUTED`, `FROZEN_LEMMA_VALID`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
