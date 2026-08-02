# Audit a limsup formalization

The intended statement has the shape `∃ A, limsup X(A) ≤ Y`. A proposed formalization has the shape `∀ A, limsup X(A) ≥ Y`.

Determine their semantic relationship. Supply two finite exact-rational model families of possible limsup values:

1. one where the intended statement is true and the proposed statement is false;
2. one where the proposed statement is true and the intended statement is false.

For each family, report the truth values of both formulas and identify a witness for the existential or a violating witness for the universal. Values must be canonical rational strings within the frozen bounds. The verifier recomputes every comparison and accepts any valid separating families. Do not claim that the underlying open problem is solved or machine verified.

Write `/app/submission.json` and bind a concise explanation at `/app/evidence/answer.txt`.

