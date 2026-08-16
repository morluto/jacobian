# Total-coloring contract audit

The frozen validator checks a proper vertex coloring and a proper edge coloring separately, then incorrectly calls their union a total coloring. Audit that contract on the supplied Petersen graph.

Submit `/app/submission.json`. Your certificate must contain:

1. a **flawed-pass assignment** using exactly the declared four-color palette whose vertex projection is proper and whose edge projection is proper, but which has at least one vertex–incident-edge color collision;
2. the complete collision list for that assignment (the order is not significant); and
3. a **repaired assignment** that is a valid total coloring of all ten vertices and fifteen indexed edges.

Both assignments must list ten vertex colors in vertex order and fifteen edge colors in the frozen edge order. The verifier independently reconstructs every vertex adjacency, edge adjacency, and incidence constraint. It accepts alternative assignments and does not trust submitted collision labels.

The conclusion is limited to this one finite validator defect and its Petersen-graph repair. It neither proves nor disproves the Total Coloring Conjecture. Claim at most `COMPUTED`; truth beyond the frozen finite instance is `NOT_ASSESSED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
