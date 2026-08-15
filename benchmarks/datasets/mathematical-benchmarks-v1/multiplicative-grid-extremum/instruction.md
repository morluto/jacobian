# Certify the maximum number of multiplicative neighbors

The frozen source asks for the maximum possible number of good pairs among 100
distinct positive integers. A pair is good when its larger member is exactly
2 or 3 times its smaller member.

Produce `/app/submission.json` matching `/app/submission_schema.json`.

Your result must contain:

- exactly 100 distinct positive integers within the declared bounds;
- the complete set of exactly 180 good index pairs `[i,j]`, with `i < j`;
- for every number at index `i`, its unique factorization
  `core * 2^two_exponent * 3^three_exponent`, where `gcd(core,6)=1`;
- the number of multiplicative components and the total numbers of nonempty
  horizontal rows and vertical columns in their exponent lattices;
- the witness projection cost `rows + columns`;
- the universal projection cost 20 and resulting edge bound 180.

For the upper bound, use the following elementary projection argument. Within
each fixed core, horizontal edges are at most the number of vertices minus the
number of occupied rows; vertical edges satisfy the analogous column bound.
A component occupying `m` exponent pairs needs row and column counts `r,c`
with `m <= r*c`. Minimize the total `r+c` over every partition of 100 vertices.
The verifier independently performs this finite minimization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
