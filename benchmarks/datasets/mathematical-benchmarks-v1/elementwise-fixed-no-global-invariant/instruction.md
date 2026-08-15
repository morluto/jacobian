# Elementwise fixed vectors without a global invariant

The offline input freezes a claim that swaps two quantifiers for finite linear
actions. Refute it by constructing a subgroup of `SL_3(F_q)` for one allowed
odd prime.

Submit two generators, the complete generated group in lexicographic matrix
order, and one nonzero fixed vector for every listed group element. The
verifier independently closes the generators under multiplication, checks
determinants, replays every fixed-vector equation, and computes the common
fixed-space intersection. The group must have order between 6 and 48, and its
common fixed space must be zero.

This is not a request to reproduce the public example. Alternative generators,
fields, conjugates, and fixed vectors are accepted whenever they satisfy the
contract. The submitted group and fixed vectors are the executable
counterexample; no prose explanation or duplicate artifact is required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
