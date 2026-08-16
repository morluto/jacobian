# Nowhere-zero 5-flow domain audit

The frozen validator checks Kirchhoff conservation modulo five but forgets that a nowhere-zero flow must exclude the zero residue on every edge.

For the oriented Petersen graph, submit:

1. a flawed-pass modular flow with exactly one zero edge and fourteen nonzero edges;
2. the complete ten-vertex signed balance vector and the zero edge index; and
3. a repaired nowhere-zero 5-flow, again with the complete balance vector.

Flows are fifteen integers in the frozen edge order, each in `0..4`. For an edge `(u,v)`, its value contributes positively at `u` and negatively at `v`; every balance must be zero modulo five. The verifier independently recomputes balances, zero support, and all domain constraints. Alternative valid flows are accepted.

This finite Assurance audit demonstrates one contract defect and one Petersen-graph repair. It does not establish Tutte's 5-Flow Conjecture. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
