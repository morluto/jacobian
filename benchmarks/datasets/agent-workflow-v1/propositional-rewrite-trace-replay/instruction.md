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
must copy `result` and `limitations` exactly. Claim at most `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** one of `TARGET_REACHED`, `NO_VALID_TRACE`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** one of `COMPLETE`, `PARTIAL`, `UNKNOWN`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/rewrite-trace.json`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `application/json`.
- **Required artifact filenames:** `evidence/rewrite-trace.json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
