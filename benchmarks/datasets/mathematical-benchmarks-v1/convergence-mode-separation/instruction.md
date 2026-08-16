# Separate convergence in probability from almost-sure convergence

On `[0,1)` equipped with Lebesgue measure, consider the dyadic typewriter sequence. At level `k`, enumerate the indicators of all half-open intervals `[j/2^k,(j+1)/2^k)` in order, using sequence index `2^k+j`.

Certify that these indicators converge to zero in probability but not almost surely. Submit exact block summaries for every frozen level, the general event-mass formula, and at least three freely chosen rational probe points in `[0,1)`. Represent rationals as integer `numerator`/positive integer `denominator` objects. Equivalent encodings such as `2/6` and `1/3` are accepted after exact `Fraction` normalization. For each probe, give the unique hit index at every level. The verifier derives the mode separation from the submitted formula, level masses, and probe traces.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
