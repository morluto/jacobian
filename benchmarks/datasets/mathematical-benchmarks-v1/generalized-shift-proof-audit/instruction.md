# Audit a generalized-shift complexity proof

The frozen input records four claims from a generated proof of unbounded shift
complexity. Diagnose all four claims by submitting exact, independently
replayable certificates:

1. a bounded-displacement map whose associated generalized shift is not
   unitary;
2. the exact operator norm of the stated scaled Fourier block;
3. the valid direction relating operator and Hilbert--Schmidt norms, together
   with a diagonal matrix showing why the claimed reverse lower bound fails;
4. an integer `m >= 2` showing that the claimed real expression
   `sqrt(1-m)` is not defined.

The first certificate may use any two distinct domain indices that collide
under `n + alpha(n)`. The Fourier size may be any odd integer from 3 through
15. The norm counterexample may use any rational diagonal entries satisfying
the required strict inequality. The verifier recomputes every certificate and
does not trust the generated solution.

This task audits the proof trace only. It does not determine the true
shift-complexity supremum.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
