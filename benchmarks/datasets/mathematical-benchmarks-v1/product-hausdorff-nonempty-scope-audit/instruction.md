# Audit the missing nonempty-factor hypothesis

The frozen ProofNetVerif prediction omits the source theorem's assumption that every factor is nonempty. Produce a complete finite topological countermodel.

Use exactly three factors. Factor 0 must have 4–7 points and a submitted topology with at least five open sets; it must be T0 but not Hausdorff. Factor 1 must be empty. Factor 2 must be a nonempty finite set of size 2–5 (its discrete topology is implicit). List every open set of factor 0 in canonical increasing order.

The verifier will independently check the topology axioms, T0 separation, failure of Hausdorff separation, the empty product cardinality, and the missing hypothesis. Submit the digest-bound task-specific witness file and

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/product-hausdorff-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
