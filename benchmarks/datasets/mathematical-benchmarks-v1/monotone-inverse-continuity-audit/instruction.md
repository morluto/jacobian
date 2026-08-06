# Audit strict monotonicity without continuity

The frozen source removes continuity from a theorem about a strictly increasing
function on a compact interval.  Within the declared two-branch rational family,
construct a strictly increasing function with a jump at zero and certify that its
image omits a rational point lying strictly between its endpoint values.

Submit canonical rational parameters, the four boundary values listed in the
schema, and any canonical rational gap witness.  The witness must lie strictly
between the left limit and the right value at zero.  This simultaneously refutes
the claimed interval-image conclusion and the existence of a two-sided inverse
on the full endpoint interval.  The verifier independently checks the parameter
bounds, strict monotonicity across both branches and the jump, every submitted
value, and the omitted-image witness.

Write `submission.json` according to `submission_schema.json`. Put a concise
derivation in `evidence/answer.txt` and bind exactly that file with its SHA-256
digest. The derivation must explain all four steps: both affine branches are
strictly increasing because their slopes are positive; the positive jump makes
cross-branch comparisons strict; the image is the union of the two branch
ranges with a missing gap; and the gap witness lies strictly between the left
limit and the right breakpoint value, so it has no preimage and the claimed
full-interval image or two-sided inverse fails. Equivalent mathematical wording
is accepted. Include one line beginning `RESULT_JSON:` followed by a JSON object
whose parsed value exactly matches the complete submitted `result` object; this
machine-readable line is part of the public evidence contract and binds the
derivation to the witness. The maximum permitted assurance is `COMPUTED`; this
task does not run a proof assistant or certify arbitrary real functions outside
the declared family.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions. Evidence prose must explain that both branches are strictly increasing, the positive jump creates an omitted image gap, and the gap witness has no preimage, so the full-interval inverse conclusion fails; RESULT_JSON binds the submitted values but does not replace this derivation.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `CONTINUITY_IS_NECESSARY`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
