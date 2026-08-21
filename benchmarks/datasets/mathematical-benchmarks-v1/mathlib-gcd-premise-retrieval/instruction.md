# Retrieve a Lean premise for a gcd goal

The frozen input contains one Lean theorem goal and the local identifiers that
are available at the proof point. Complete the goal with one `exact`
declaration application.

Submit the fully qualified declaration identifier and its ordered argument
identifiers. The hidden verifier inserts that application into the frozen Lean
source and elaborates it with Lean 4.31.0. Any declaration application that
elaborates is accepted; no preferred theorem spelling or tool workflow is
required.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit one fully qualified Lean declaration and the ordered local identifiers passed to it. The verifier accepts any exact declaration application that elaborates against the frozen goal.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
