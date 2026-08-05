# Repair the modular irreducibility step

The frozen proof says that `f(x)=x^4-4x+1` is irreducible over `Q` because its
reduction `x^4+1` is irreducible over `F_2`. Audit that claim, give the exact
factorization at the bad prime, and repair the step using a freely chosen prime
`3 <= p <= 97` for which the reduction of `f` is irreducible.

For the repair, submit the coefficient vectors (constant term first) of
`x^(p^2) mod f` and `x^(p^4) mod f`. The verifier independently checks primality,
modular polynomial arithmetic, the degree-four Rabin conditions
`gcd(f,x^(p^2)-x)=1` and `x^(p^4)-x=0 mod f`, and the bad-prime factorization.

Evidence must be a regular file no larger than 1 MiB, contain exactly one
`RESULT_JSON:` line equal to `result`, and provide at least 140 characters
explaining why the original implication is invalid and why the replacement
proves irreducibility over `Q`. Do not claim proof-assistant verification.

Use this exact limitations entry: `The verifier does not assess the source
proof's later Galois-group or density claims and does not invoke a proof
assistant.`

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, scope, completeness, evidence binding, and assurance as separate protocol dimensions.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `BAD_REDUCTION_DIAGNOSED_AND_IRREDUCIBILITY_REPAIRED`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
