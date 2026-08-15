# Audit strict monotonicity without continuity

The frozen source removes continuity from a theorem about a strictly increasing
function on a compact interval.  Within the declared two-branch rational family,
construct a strictly increasing function with a jump at zero and certify that its
image omits a rational point lying strictly between its endpoint values.

Submit canonical rational parameters, the four boundary values listed in the
schema, and any canonical rational gap witness.  The witness must lie strictly
between the left limit and the right value at zero.  This simultaneously refutes
the claimed interval-image conclusion and the existence of a two-sided inverse
on the full endpoint interval.  The verifier independently checks the parameter
bounds, strict monotonicity across both branches and the jump, every submitted
value, and the omitted-image witness.

Write `submission.json` according to `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
