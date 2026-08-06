# Audit the missing nonempty-factor hypothesis

The frozen ProofNetVerif prediction omits the source theorem's assumption that every factor is nonempty. Produce a complete finite topological countermodel.

Use exactly three factors. Factor 0 must have 4–7 points and a submitted topology with at least five open sets; it must be T0 but not Hausdorff. Factor 1 must be empty. Factor 2 must be a nonempty finite set of size 2–5 (its discrete topology is implicit). List every open set of factor 0 in canonical increasing order.

The verifier will independently check the topology axioms, T0 separation, failure of Hausdorff separation, the empty product cardinality, and the missing hypothesis. Submit the digest-bound evidence file and claim only `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Complete finite-topology replay; Lean elaboration and infinite spaces remain outside scope.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `EMPTY_FACTOR_MASKS_NON_HAUSDORFF_FACTOR`
- **Assurance:** scoreable values are `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/product-hausdorff-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/product-hausdorff-audit.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
