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

Write `submission.json` to the supplied schema. Write
`evidence/resolution-proof.json` with exactly `schema_version`, `task_id`,
`result`, and `limitations`, copy the corresponding submission values exactly,
and bind it by SHA-256. Claim at most `COMPUTED` and use limitation code
`FROZEN_RESOLUTION_CALCULUS_ONLY`.
