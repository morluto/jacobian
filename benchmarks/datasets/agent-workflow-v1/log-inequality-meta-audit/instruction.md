# Audit a proof, its evaluation, and the meta-evaluation

Audit every layer of the frozen conversation trace. Determine the truth of the
universal inequality, validate or reject the proposed counterexample, assess
the score-zero evaluation under its stated instruction-following rubric, and
assess the meta-evaluation. Keep mathematical correctness separate from whether
the response followed the original request to prove the claim.

Return the exact algebraic comparison certificate requested by the schema.
Write `submission.json` to the exact agent-visible `submission_schema.json`.
Put a concise audit in `evidence/answer.txt`, and bind that file with its
SHA-256 digest. The audit must state the Pythagorean counterexample, the exact
integer-power comparison, the false universal conclusion, and why score zero
matches the instruction-following rubric.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `CLAIM_FALSE_EVALUATION_RUBRIC_CONSISTENT`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
