# Audit a strengthened metric conclusion

The frozen natural theorem says that disjoint closed subsets of a metric space are separated. A
candidate Lean statement instead asserts that there is one positive `epsilon` bounding every
cross-distance from below.

Diagnose this semantic strengthening by constructing two disjoint locally finite subsets of the
rational plane whose distance infimum is zero. Choose a start index from 4 through 20 and submit
eight consecutive indexed point pairs. For every row use `A_n = (n,0)` and `B_n = (n,1/n)`, with
each exact rational coordinate and distance represented as an integer `numerator` and positive
integer `denominator`. Equivalent encodings such as `2/8` and `1/4` are accepted after
normalization. Also submit four to eight distinct positive
rational epsilons in strictly decreasing order, each paired with an index `N` such that
`N` is at least the start index and `1/N < epsilon`; indices must strictly increase.

Identify the natural and predicted conclusions, their semantic relationship, the missing assumption,
and the local-finiteness rule that makes both infinite sets closed.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
