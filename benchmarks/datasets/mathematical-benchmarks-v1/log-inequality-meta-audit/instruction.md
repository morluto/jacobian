# Audit a proof, its evaluation, and the meta-evaluation

Audit every layer of the frozen conversation trace. Determine the truth of the
universal inequality, validate or reject the proposed counterexample, assess
the score-zero evaluation under its stated instruction-following rubric, and
assess the meta-evaluation. Keep mathematical correctness separate from whether
the response followed the original request to prove the claim.

Return the exact algebraic comparison certificate requested by the schema.
Write `submission.json` to the exact agent-visible `submission_schema.json`.
The structured result must state the Pythagorean counterexample, the exact
integer-power comparison, the false universal conclusion, and why score zero
matches the instruction-following rubric.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
