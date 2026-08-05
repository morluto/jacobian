# Repair a triplewise-empty extremal bound

The frozen source claims the maximum family size is `2n`. Audit that argument for distinct subsets, including the empty set.

Submit the corrected general formula, the incidence-budget facts used for the upper bound, and extremal families on ground sets of sizes 7, 8, and 11. Represent each subset as a sorted list of zero-based elements. The verifier independently checks uniqueness, range, triplewise-empty intersections, element frequencies, and the claimed maximum size.

The general symbolic incidence argument is inspected through its typed certificate but is not machine-proved for every `n`; assurance remains `COMPUTED`.

Write one JSON submission to `/app/submission.json` using the schema in
`/app/submission_schema.json`. Include one digest descriptor for the evidence
file at `/app/evidence/answer.txt`. That text must state the corrected formula,
the at-most-two element-frequency bound, the at-least-two incidence cost for
each remaining non-singleton, and that the displayed constructions attain the
bound. Disclose at least one limitation describing the finite probes and the
`COMPUTED` assurance ceiling.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `SOURCE_BOUND_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
