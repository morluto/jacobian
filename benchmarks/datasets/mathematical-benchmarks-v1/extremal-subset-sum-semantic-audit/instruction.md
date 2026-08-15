# Audit an extremal subset-sum formalization

Compare the frozen informal requirement with the proposed Lean declaration.
Determine whether the declaration faithfully preserves the fixed outer
parameter and the requirement that no subset of a candidate sums to the target.

Supply two exact certificates:

1. two cutoff multipliers for the same target whose legacy extrema disagree,
   showing that the shadowed universal binder makes one function value satisfy
   incompatible equations;
2. the exact legacy and intended extrema on the frozen finite universe,
   including a legacy-optimal candidate, an intended-optimal candidate, and a
   subset that invalidates the legacy candidate under the intended predicate.

The legacy predicate checks only the sum of the whole candidate. The intended
predicate checks every subset, including the empty subset and the candidate
itself. Use lists as mathematical sets: entries must be strictly increasing.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
