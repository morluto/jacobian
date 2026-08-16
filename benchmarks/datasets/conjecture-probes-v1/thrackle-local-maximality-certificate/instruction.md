# Exact local thrackle certificate

On the five frozen rational points, select exactly five of the ten candidate straight-line edges so every pair of selected edges meets exactly once: either at a shared endpoint or in one proper interior crossing.

Submit the sorted selected edges, a complete classification of all ten selected-edge pairs (`SHARED_ENDPOINT` or `PROPER_CROSSING`), and, for each excluded candidate edge, the lexicographically first selected edge disjoint from it. The latter witnesses that no excluded edge can be added while preserving the thrackle property.

The verifier independently evaluates orientation determinants, endpoint incidence, every pair classification, and every local-maximality witness. Collinear overlap and mere line intersection outside the closed segments do not count. Edge and witness order is canonical.

This Regression certificate proves only a five-edge thrackle locally maximal inside this frozen `K5` candidate universe. It does not prove Conway's Thrackle Conjecture.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
