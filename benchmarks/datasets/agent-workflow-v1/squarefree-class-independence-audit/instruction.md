# Audit the squarefree-class argument

Prove the frozen universal claim by connecting three layers rather than by constructing one example set:

1. classify positive integers by their squarefree kernel and establish exactly when a product is a square;
2. translate the ordered-pair count into a sum of squares of class sizes and an independent transversal into distinct classes;
3. give a complete modular certificate showing that `2023` cannot be a sum of at most three integer squares.

You may choose any modulus within the frozen bounds. Submit its complete, sorted set of quadratic residues and the exact target residue. The verifier will independently enumerate all zero-, one-, two-, and three-square residue sums; checking only selected decompositions is insufficient.

Write `/app/submission.json` matching the supplied schema. Bind one concise derivation at `/app/evidence/answer.txt`. Do not claim machine verification or a classification beyond the frozen theorem.

Include one line beginning `RESULT_JSON:` in the evidence file, followed by the JSON serialization of the submitted `result` object. The derivation must state the squarefree-kernel product equivalence, the class-size count identity, the distinct-class transversal step, and the concrete modular certificate.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `FOUR_ELEMENT_INDEPENDENT_SUBSET_FORCED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
