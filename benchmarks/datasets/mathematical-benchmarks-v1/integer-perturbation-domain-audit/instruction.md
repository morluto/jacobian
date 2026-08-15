# Audit a perturbation-domain mismatch

The informal definition quantifies over bounded **integer-valued** perturbations,
but the frozen formal contract restricts `b` to natural numbers.

Submit a two-part semantic certificate. First, record the symbolic lower-bound
argument showing that for natural `a,b`, the hypotheses `a >= 0` and `b != 0`
force `a+b >= 1`; therefore the separate `a+b != 0` hypothesis is redundant.
Second, construct a periodic integer-valued pair of sequences within the frozen
bounds. Every `a` value must be positive, every `b` value nonzero, `b` must take
both signs, and at least two indices must have `a+b=0`. Report every period
value, exact sum, extremal `b` value, and cancellation index.

The verifier independently recomputes all bounds and periodic values. Write the
exact schema to `submission.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
