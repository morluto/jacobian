# Nowhere-zero 5-flow domain audit

The frozen validator checks Kirchhoff conservation modulo five but forgets that a nowhere-zero flow must exclude the zero residue on every edge.

For the oriented Petersen graph, submit:

1. a flawed-pass modular flow with exactly one zero edge and fourteen nonzero edges;
2. the complete ten-vertex signed balance vector and the zero edge index; and
3. a repaired nowhere-zero 5-flow, again with the complete balance vector.

Flows are fifteen integers in the frozen edge order, each in `0..4`. For an edge `(u,v)`, its value contributes positively at `u` and negatively at `v`; every balance must be zero modulo five. The verifier independently recomputes balances, zero support, and all domain constraints. Alternative valid flows are accepted.

This finite Assurance audit demonstrates one contract defect and one Petersen-graph repair. It does not establish Tutte's 5-Flow Conjecture. Claim at most `CHECKED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Exact nonzero-domain audit and repair for one Petersen-graph modular flow only.

The evidence file must be a JSON object with exactly four fields: `schema_version` must be `"1"`, `task_id` must equal the submission task ID, `result` must be an exact JSON copy of `submission.json`'s result object, and `limitations` must be an exact JSON copy of its limitations list.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `CONSERVATION_ONLY_IS_UNSOUND_AND_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED`, `CHECKED` (ceiling `CHECKED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/answer.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
