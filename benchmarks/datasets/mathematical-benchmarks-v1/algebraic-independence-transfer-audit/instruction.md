# Audit an algebraic-independence transfer

The frozen proof cites an external theorem that `delta,Ddelta,D2delta` are algebraically independent, then compresses the remaining transfer to `P,Q,R` into several “so” statements. Treat the cited theorem as a premise; repair only the exact algebraic transfer.

Represent every polynomial as an unordered list of distinct sparse terms. A term is `{"coefficient":{"numerator":a,"denominator":b},"exponents":[i,j,k]}` in the variable order named by its field. Equivalent encodings such as `2/2` and `1` are accepted after exact `Fraction` normalization; zero terms and duplicate exponent vectors are forbidden.

Submit the exact rational definitions

- `P = Ddelta/delta` and `Q = (13 Ddelta^2 - 12 delta D2delta)/delta^2` in source order `(delta,Ddelta,D2delta)`;
- the inverse formulas `Ddelta=delta*P` and `D2delta=delta*(13P^2-Q)/12` in order `(delta,P,Q)`;
- `S=Q^3-delta` in order `(delta,P,Q)` and its inverse `delta=Q^3-S` in order `(P,Q,S)`.

Finally compute the frozen conjugate norm `F(P,Q,R)F(P,Q,-R)` as a polynomial in `(P,Q,S)` with `S=R^2`. This certifies only the displayed coordinate identities and this one exact norm calculation. It does **not** certify the universal nonzero-norm argument for an arbitrary polynomial relation, repair the general algebraic-independence proof, or prove the external transcendence premise.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
