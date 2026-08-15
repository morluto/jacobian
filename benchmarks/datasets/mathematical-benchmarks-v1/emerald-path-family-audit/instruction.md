# Audit the claimed uniqueness of an APMO path construction

The frozen ProofBench model response concludes that the only possible parameters are `alpha=beta=1`. Audit that conclusion.

Submit a nontrivial positive rational pair with `alpha >= beta`, `alpha+beta=2`, and `alpha != beta`. Equivalent rational spellings are accepted. Rational numerators and denominators use ordinary signed base-10 notation with at most 64 digits each; exponent notation is not accepted. Use the path `x_n=ceil(n/2)`, `y_n=floor(n/2)`. Provide an exact parity certificate showing that `floor(x_n alpha+y_n beta)=n` for every nonnegative integer `n`, and a 16-row exact trace.

The digest-bound `evidence/answer.txt` must contain exactly six nonempty lines: `emerald-path-family-certificate-v1`, the four fields `alpha:`, `beta:`, `even_offset:`, and `odd_offset:` using the submitted strings, then `trace_sha256:` followed by the SHA-256 of the submitted trace serialized as sorted-key compact JSON.


<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result and validates the declared task-specific witness.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
