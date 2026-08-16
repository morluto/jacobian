# Negative control for a polynomial-map collision

The `claimed_image` in `input.json` has been mutated away from the true common image. Submit the finite collision object: the two distinct points, their exact images under the polynomial map, and confirm that the images are equal. The verifier replays the polynomial map from the frozen input and checks the collision. The truth values — whether both points map to the claimed image and whether non-invertibility holds — are derived by the evaluator from the submitted collision object.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
