# Retrieve premises and reconstruct a quotient-group proof

From the frozen candidate library, select exactly the external premises needed
to prove that a quotient of an abelian group by an arbitrary subgroup is
abelian. Then submit a proof DAG using the registered rule vocabulary in the
input.

Every step must name its rule, list already available input facts, and produce
one declared output fact. The verifier replays the DAG, rejects circular or
unjustified steps, and rejects unnecessary selected premises. Write
`submission.json` to the exact agent-visible `submission_schema.json`. Put a
concise proof explanation in `evidence/answer.txt`, and bind that file with its
SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `QUOTIENT_GROUP_IS_ABELIAN`, `PROOF_NOT_FOUND`, `UNSUPPORTED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
