Construct a strictly positive continuous function on `[1,+infinity)` for which
the improper integral diverges but the series of integer samples converges.

The function is a triangular-tent construction:

```
f(x) = x^(-2) + sum_{n>=1} max(1 - |x - (n + 1/2)| / (alpha/n), 0)
```

where `0 < alpha <= 1/4` is a rational parameter. Each spike is a triangular
tent of height `1` centered at `(n + 1/2)` with half-width `alpha/n`. The
spike area equals the half-width. The baseline power is `2` (from `x^(-2)`).

Submit the exact first twelve spike supports and areas, the twelve integer
samples, and the parameter `alpha`. Each spike support must be disjoint from
every other support and must avoid every integer. The spike areas must form a
divergent series while the integer samples form a convergent series.

Represent each rational field as an integer `numerator` and positive integer
`denominator`. Equivalent encodings such as `2/8` and `1/4` are accepted after exact `Fraction` normalization.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
