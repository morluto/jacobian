# Replay independent-set transfer states on square grids

For each `n` from 2 through 5, count black/white colorings of an `n × n` grid with no horizontally or vertically adjacent black cells.

Use row masks. Submit, for every `n`, the sorted list of horizontally valid masks, the number of ordered vertically compatible mask pairs, the total number of partial colorings after each successive row, and the final count `x_n`. Also submit `x_2+x_3+x_4+x_5`. The four cases may appear in any order in the `cases` array; each case is matched by its `n` value.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
