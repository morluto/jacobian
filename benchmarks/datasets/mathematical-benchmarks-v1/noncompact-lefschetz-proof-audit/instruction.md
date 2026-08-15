# Audit a noncompact Lefschetz argument

The frozen input contains a lemma used by two model answers: for a proper map of
a boundaryless manifold with finite-dimensional compact-support cohomology, no
fixed points supposedly imply zero compact-support Lefschetz number.

Audit that lemma and its use in the downstream torsion-freeness proof.

1. Give a nonzero rational translation of the real line.
2. Record the fixed-point equation and show it has no solution.
3. Record the compact-support cohomology dimensions, induced top-degree action,
   and exact compact-support Lefschetz number.
4. Identify the invalid compact-support step in the frozen graph/diagonal proof.
5. State precisely what the counterexample invalidates and what it does not
   decide about the original First Proof research question.

Write `submission.json` to the exact agent-visible schema.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
