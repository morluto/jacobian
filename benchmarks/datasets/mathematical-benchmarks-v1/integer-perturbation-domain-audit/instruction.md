# Audit a perturbation-domain mismatch

The informal definition quantifies over bounded **integer-valued** perturbations,
but the frozen formal contract restricts `b` to natural numbers.

Submit a two-part semantic certificate. First, record the symbolic lower-bound
argument showing that for natural `a,b`, the hypotheses `a >= 0` and `b != 0`
force `a+b >= 1`; therefore the separate `a+b != 0` hypothesis is redundant.
Second, construct a periodic integer-valued pair of sequences within the frozen
bounds. Every `a` value must be positive, every `b` value nonzero, `b` must take
both signs, and at least two indices must have `a+b=0`. Report every period
value, exact sum, extremal `b` value, and cancellation index.

The verifier independently recomputes all bounds and periodic values. Do not
claim Lean compilation or any irrationality theorem. State both limitations in
the `limitations` field. Write the exact schema to `submission.json`; put a
concise audit and a matching `RESULT_JSON:` line in `evidence/answer.txt`, and
bind its SHA-256 digest.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier checks the declared result, exact domain-semantics scope, completeness, digest-bound evidence, limitation claims, and assurance as separate protocol dimensions. State both that Lean compilation is not assessed and that no irrationality theorem is claimed.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
