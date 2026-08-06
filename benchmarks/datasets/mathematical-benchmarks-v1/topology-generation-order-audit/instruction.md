# Audit an order-sensitive topology formalization

The frozen input contrasts an informal generated-topology objective with a proposed common-bound contract. Construct a finite model, using the declared bitmask semantics, that determines whether the proposed contract captures the source objective.

Submit `/app/submission.json` following `/app/submission_schema.json` and digest-bound prose at `/app/evidence/answer.txt`.

Your model must use three or four points and contain at least two distinct, incomparable, nontrivial input topologies. Submit:

- each input topology as a sorted list of subset bitmasks;
- the unique smallest topology containing every submitted input topology;
- a topology contained in every input topology but distinct from the generated topology;
- one open-set bitmask witnessing the failed containment direction;
- the semantic diagnosis.

The evidence prose must explain why the two topology constructions satisfy different order contracts. Do not claim Lean compilation or proof-assistant verification. Claim `COMPUTED` assurance and complete scope.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier exhaustively adjudicates the submitted finite topology model under ordinary family inclusion.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `PROPOSED_CONTRACT_TOO_WEAK`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
