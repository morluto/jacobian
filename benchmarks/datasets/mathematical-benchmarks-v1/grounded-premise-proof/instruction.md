# Retrieve premises and reconstruct a quotient-group proof

From the frozen candidate library, select exactly the external premises needed
to prove that a quotient of an abelian group by an arbitrary subgroup is
abelian. Then submit a proof DAG using the registered rule vocabulary in the
input.

Every step must name its rule, list already available input facts, and produce
one declared output fact. The verifier replays the DAG, rejects circular or
unjustified steps, and rejects unnecessary selected premises. Write
`submission.json` to the exact agent-visible `submission_schema.json`. Put a
task-specific witness in `evidence/answer.txt`, and bind that file with its
SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
