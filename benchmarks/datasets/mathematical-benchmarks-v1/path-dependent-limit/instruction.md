# Certify a limit missed by every straight line

Construct a function from the family

`f_p(x,y) = x^(2p) y / (x^(4p) + y^2)`, with `1 <= p <= 5`, away from the
origin and define its value at the origin to be zero.

Submit an exact certificate showing that the limit is zero along every
straight line through the origin, including both coordinate axes, while the
two-variable limit does not exist. Your all-lines certificate must give the
orders obtained after substituting `y=m*x` for an arbitrary nonzero slope and
explain why the resulting quotient tends to zero. Also submit three distinct,
freely chosen nonzero rational parameters `c` for paths `y=c*x^(2p)`, with the
exact nonzero limit on each path.

The verifier independently checks the exponent relations and every rational
path value. Write `c` and each exact limit as a signed integer, finite decimal,
or signed fraction such as `+1`, `0.5`, or `-2/3`; fraction denominators must be
nonzero. Numerical sampling or a conclusion label alone is insufficient.
The evidence must contain exactly one `RESULT_JSON:` line equal to `result` and
must include an actual explanation that agreement on every straight line does
not establish the multivariable limit, rather than a keyword list. Include a
limitation stating that the verifier does not establish the real-limit
statements in a proof assistant. Do not claim proof-assistant verification.
The digest-bound evidence file must be no larger than 1 MiB. Full reward
requires `COMPUTED`; `UNVERIFIED` is diagnostic-only.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `ALL_LINEAR_PATHS_ZERO_BUT_LIMIT_DOES_NOT_EXIST`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
