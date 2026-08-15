# Replay a local propositional rewrite trace

Starting from `initial_ast` in `input.json`, reach `target_ast` using only the
registered local rules. Each step must give the rule name, a zero-based child
path into the current AST, and the complete AST after that one rewrite. Paths
refer to `args`; atoms and `false` have no children.

Rules may be applied in any valid order. `FLATTEN_ASSOCIATIVE` flattens only
direct children with the same `and` or `or` operator. `CONTRADICTION` replaces
an `and` node containing both a subtree and its direct negation by `false`.
Do not jump between merely equivalent formulas: every submitted transition is
replayed exactly.

Write `submission.json` and digest-bind `evidence/rewrite-trace.json`, which
must copy `result` exactly.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/rewrite-trace.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
