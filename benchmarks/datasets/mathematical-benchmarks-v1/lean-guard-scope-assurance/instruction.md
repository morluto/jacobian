# Audit Lean semantic-guard findings across scope and pathological types

For every frozen case in `/app/input.json`, determine the exact semantic-trap
findings that a sound checker should emit.  Preserve the input order and attach
the controlling reason code for each decision.

source repository's expected messages.  Your audit must distinguish full proof
state scope from textual binder order, ordinary numeric types from a custom
type where `0` and `1` are definitionally equal, proof terms from theorem
statements, and division-by-zero risk from precision-loss truncation.

The offline verifier derives every expected finding from the frozen semantic facts.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result. Return exactly five cases in the order given in /app/input.json, preserving each case's id.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
