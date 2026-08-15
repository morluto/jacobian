# Audit the summation domain

For fixed `n`, the source defines `a_n` by summing over every binary function
on the positive integers with finite support, then silently replaces that set
by subsets of `{1,...,n}`.

Choose `4 <= n <= 12` and provide at least six distinct singleton supports
strictly beyond `n`. Compute each summand and the resulting finite partial-sum
lower bound. Then repair the definition by restricting supports to
`{1,...,n}`: provide at least three exact rational checkpoints for

`c_n = product_{k=1}^n (2+1/k^2) / n!`

and a uniform ratio certificate showing `c_{n+1}/c_n <= 3/4` for every `n>=2`.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/scope-audit.json`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `application/json`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
