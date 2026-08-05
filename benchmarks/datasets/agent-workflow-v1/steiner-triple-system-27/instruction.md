# Construct a Steiner triple system of order 27

Construct a collection of exactly 117 three-element blocks on the labeled point
set `{0,...,26}` such that every unordered pair of distinct points occurs in
exactly one block. Submit the complete block collection; block and collection
order do not matter.

The verifier independently canonicalizes the blocks, counts all 351 unordered
pairs, and requires multiplicity exactly one. It accepts any valid labeled
Steiner triple system, not only an affine-space construction or the hidden
Oracle design.

Write `submission.json` according to `submission_schema.json`. The bound
`evidence/answer.txt` must contain exactly these four nonblank lines, with the
placeholders replaced by values from your submitted result:

```text
steiner-triple-system-certificate-v1
result_sha256: <SHA-256 of the sorted-key compact result JSON>
order: <submitted order>
block_count: <number of submitted blocks>
```

The assurance ceiling is `COMPUTED`: the finite design is exhaustively checked,
but the general source theorem is not machine-proved.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier reports mathematical, evidence, input-binding, scope, and assurance dimensions separately.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `STEINER_TRIPLE_SYSTEM_CONSTRUCTED`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
