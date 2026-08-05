# Audit a limsup formalization

The intended statement has the shape `∃ A, limsup X(A) ≤ Y`. A proposed formalization has the shape `∀ A, limsup X(A) ≥ Y`.

Determine their semantic relationship. Supply two finite exact-rational model families of possible limsup values:

1. one where the intended statement is true and the proposed statement is false;
2. one where the proposed statement is true and the intended statement is false.

For each family, report the truth values of both formulas and identify a witness for the existential or a violating witness for the universal. Values must be canonical rational strings within the frozen bounds. The verifier recomputes every comparison and accepts any valid separating families.

Write `/app/submission.json` and bind a concise explanation at `/app/evidence/answer.txt`. The explanation must agree with the submitted result: include exactly one `RESULT_JSON:` line whose JSON equals the submitted `result` object, and use the words `existential`, `universal`, and `incomparable` to describe the relationship and the two separating models. Do not claim that the underlying open problem is solved or machine verified; a limitation must state this restriction in unambiguous negated language rather than merely mentioning the open problem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `FORMULAS_NOT_EQUIVALENT`, `INSUFFICIENT_EVIDENCE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
