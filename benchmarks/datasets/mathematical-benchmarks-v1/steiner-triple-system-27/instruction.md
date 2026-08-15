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

but the general source theorem is not machine-proved.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
