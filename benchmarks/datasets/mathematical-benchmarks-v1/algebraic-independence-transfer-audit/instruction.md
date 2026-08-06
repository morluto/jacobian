# Audit an algebraic-independence transfer

The frozen proof cites an external theorem that `delta,Ddelta,D2delta` are algebraically independent, then compresses the remaining transfer to `P,Q,R` into several “so” statements. Treat the cited theorem as a premise; repair only the exact algebraic transfer.

Represent every polynomial as an unordered list of distinct sparse terms. A term is `{"coefficient":"a/b","exponents":[i,j,k]}` in the variable order named by its field. Coefficients must be canonical rationals; zero terms and duplicate exponent vectors are forbidden.

Submit the exact rational definitions

- `P = Ddelta/delta` and `Q = (13 Ddelta^2 - 12 delta D2delta)/delta^2` in source order `(delta,Ddelta,D2delta)`;
- the inverse formulas `Ddelta=delta*P` and `D2delta=delta*(13P^2-Q)/12` in order `(delta,P,Q)`;
- `S=Q^3-delta` in order `(delta,P,Q)` and its inverse `delta=Q^3-S` in order `(P,Q,S)`.

Finally compute the frozen conjugate norm `F(P,Q,R)F(P,Q,-R)` as a polynomial in `(P,Q,S)` with `S=R^2`. This certifies only the displayed coordinate identities and this one exact norm calculation. It does **not** certify the universal nonzero-norm argument for an arbitrary polynomial relation, repair the general algebraic-independence proof, or prove the external transcendence premise.

The evidence file `evidence/answer.txt` must state the following three facts in the solver's own words: (1) the first and second coordinate changes are birational (with the displayed inverse formulas); (2) the conjugate norm is computed exactly over the rationals (QQ); (3) the modular-form independence theorem remains a trusted premise. Additional derivation content is allowed and ignored.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit ten exact sparse polynomials forming the two forward/inverse transfer stages and the quadratic conjugate norm. Term order is free, but coefficients and monomials are canonicalized independently. In the solver's own words, the limitations array must disclose that the external algebraic-independence theorem is a trusted premise not verified here.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `EXPLICIT_TRANSFER_CHAIN_REPAIRS_COMPRESSED_PROOF`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
