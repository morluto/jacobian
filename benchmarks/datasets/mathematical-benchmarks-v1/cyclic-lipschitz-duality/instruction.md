# Certify a cyclic Lipschitz optimum

Read `/app/input.json`. Maximize the sum at the marked positions over real cyclic sequences satisfying the zero-sum and adjacent-difference constraints.

Write a task-specific JSON witness at `evidence/answer.txt` with `schema_version: "1"`, the task ID, the primal and dual certificates below, and `optimality: "weak_duality_after_median_minimum"`. Give a canonical rational feasible sequence and a canonical rational edge circulation `q`. With indices modulo the frozen cycle size, require `q_i-q_(i-1)=w_i`, where `w_i=1-m/n` at a marked position and `-m/n` otherwise, for cycle size `n` and `m` marked positions. Its `L1` norm is the dual value.



<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
