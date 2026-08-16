# Retrieve premises and reconstruct a quotient-group proof

From the frozen candidate library, select exactly the external premises needed
to prove that a quotient of an abelian group by an arbitrary subgroup is
abelian. Then submit a proof DAG using the registered rule vocabulary in the
input.

Every step must name its rule, list already available input facts, and produce
one declared output fact. The verifier replays the DAG, rejects circular or
unjustified steps, and rejects unnecessary selected premises. Write
`submission.json` to the exact agent-visible `submission_schema.json`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
