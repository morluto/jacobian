# Construct a Steiner triple system of order 27

Construct a collection of exactly 117 three-element blocks on the labeled point
set `{0,...,26}` such that every unordered pair of distinct points occurs in
exactly one block. Submit the complete block collection; block and collection
order do not matter.

The verifier independently canonicalizes the blocks, counts all 351 unordered
pairs, and requires multiplicity exactly one. It accepts any valid labeled
Steiner triple system, not only an affine-space construction or the hidden
Oracle design.

Write `submission.json` according to `submission_schema.json`. The construction
is a finite combinatorial certificate; the general source theorem is not
machine-proved.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
