# Certify a cyclic Lipschitz optimum

Read `/app/input.json`. Maximize the sum at the marked positions over real cyclic sequences satisfying the zero-sum and adjacent-difference constraints. The scope description must identify the frozen cycle size and every marked index; equivalent wording and ordering are accepted.

Submit the standard envelope. Give a canonical rational feasible sequence and a canonical rational edge circulation `q`. With indices modulo the frozen cycle size, require `q_i-q_(i-1)=w_i`, where `w_i=1-m/n` at a marked position and `-m/n` otherwise, for cycle size `n` and `m` marked positions. Its `L1` norm is the dual value.

Bind `evidence/answer.txt` to the submission. It must be a JSON object with exactly these keys: `schema_version` (`"1"`), `task_id`, `primal` (`sequence_length`, `zero_sum`, `adjacent_bound`, and `objective`), `dual` (`divergence`, `l1_cost`, and `minimum_cost`), `optimality` (`"weak_duality_after_median_minimum"`), and `limitations`. Use the derived objective and costs; set `primal.zero_sum` to `"0"`, `primal.adjacent_bound` to `"1"`, and `dual.divergence` to `"q_i-q_(i-1)=w_i"`.

The verifier derives the instance from the bound frozen input, checks primal feasibility/value, flow divergence, exact `L1` cost, and independently recomputes the minimum circulation cost from cumulative imbalances and their median. Assurance remains `COMPUTED`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `OPTIMUM_CERTIFIED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** a string value
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
