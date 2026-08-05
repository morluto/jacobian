# Audit the claimed uniqueness of an APMO path construction

The frozen ProofBench model response concludes that the only possible parameters are `alpha=beta=1`. Audit that conclusion.

Submit a nontrivial positive rational pair with `alpha >= beta`, `alpha+beta=2`, and `alpha != beta`. Equivalent rational spellings are accepted. Use the path `x_n=ceil(n/2)`, `y_n=floor(n/2)`. Provide an exact parity certificate showing that `floor(x_n alpha+y_n beta)=n` for every nonnegative integer `n`, and a 16-row exact trace.

The digest-bound `evidence/answer.txt` must contain exactly six nonempty lines: `emerald-path-family-certificate-v1`, the four fields `alpha:`, `beta:`, `even_offset:`, and `odd_offset:` using the submitted strings, then `trace_sha256:` followed by the SHA-256 of the submitted trace serialized as sorted-key compact JSON.

Include this published limitation exactly: `The certificate refutes the published singleton claim and proves sufficiency for its submitted family member; it does not independently prove necessity for every possible trip.` Do not claim `VERIFIED`.
