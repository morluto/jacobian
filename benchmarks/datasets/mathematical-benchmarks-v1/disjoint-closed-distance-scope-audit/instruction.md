# Audit disjoint closed sets versus positive distance

The frozen ProofNetVerif pair strengthens “separated” to a uniform positive-distance claim. Construct two disjoint closed subsets of the Euclidean plane that refute that strengthening.

Use a parametric pair

- `A_n = (h*n, 0)`;
- `B_n = (h*n, s/(n+c))`, for positive integer `n`;

where `2 <= h <= 20`, `1 <= s <= 20`, and `2 <= c <= 20`. Submit ten distinct sample indices and their exact squared same-index distances. For each epsilon `1/k`, `k=2,...,9`, submit an index whose distance is strictly smaller.

Also submit the exact lower bound on squared horizontal separation between distinct indices. This is the certificate that each family has no finite accumulation point and is closed. State the formalized positive-distance conclusion and the corrected conclusion separately.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
