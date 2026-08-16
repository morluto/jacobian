# Audit the missing nonempty-factor hypothesis

The frozen ProofNetVerif prediction omits the source theorem's assumption that every factor is nonempty. Produce a complete finite topological countermodel.

Use exactly three factors. Factor 0 must have 4–7 points and a submitted topology with at least five open sets; it must be T0 but not Hausdorff. Factor 1 must be empty. Factor 2 must be a nonempty finite set of size 2–5 (its discrete topology is implicit). List every open set of factor 0 in canonical increasing order.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
