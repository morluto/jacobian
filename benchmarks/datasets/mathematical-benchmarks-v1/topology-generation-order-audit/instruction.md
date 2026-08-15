# Audit an order-sensitive topology formalization

The frozen input contrasts an informal generated-topology objective with a proposed common-bound contract. Construct a finite model, using the declared bitmask semantics, that determines whether the proposed contract captures the source objective.

Submit `/app/submission.json` following `/app/submission_schema.json`.

Your model must use three or four points and contain at least two distinct, incomparable, nontrivial input topologies. Submit:

- each input topology as a sorted list of subset bitmasks;
- the unique smallest topology containing every submitted input topology;
- a topology contained in every input topology but distinct from the generated topology;
- one open-set bitmask witnessing the failed containment direction;
- the semantic diagnosis.

The semantic diagnosis must explain why the two topology constructions satisfy
different order contracts.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
