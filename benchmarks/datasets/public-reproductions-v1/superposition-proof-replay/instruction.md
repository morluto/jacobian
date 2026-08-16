# Reconstruct a first-order resolution proof

The frozen input contains eight shuffled universally quantified clauses. Some
are axioms; every other clause must be a binary resolvent of two earlier clauses.

Return a topologically ordered derivation for every non-axiom clause. Each step
must name the child and its two parents. Parent order is irrelevant. Every
clause must occur exactly once in the resulting proof graph, and the declared
root must be the frozen target clause.

The verifier independently parses and replays binary first-order resolution,
including variable standardization and unification. Clause comparison is modulo
variable renaming, literal order, duplicate literals, and equality orientation.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
