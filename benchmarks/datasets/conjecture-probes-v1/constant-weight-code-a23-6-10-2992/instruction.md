# Construct a 2992-word binary constant-weight code

Submit exactly 2992 distinct binary words of length 23. Every word must have Hamming weight 10, and every distinct pair must have Hamming distance at least 6.

Encode each word as a six-digit hexadecimal string representing its 23 coordinate bits. Leading zeroes are required; letter digits may be lowercase only. Ordering is irrelevant. The verifier parses the represented words and recomputes every weight and pairwise distance.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
