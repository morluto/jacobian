# Audit a Newton-polygon factorization lemma

The frozen input states the hypotheses and conclusion of an erroneous
factorization lemma from the first version of a paper, together with the kind
of left-edge control added in the correction.

Construct two nonconstant integer polynomials whose product satisfies every
old hypothesis at submitted indices `ell < j`, while both factor constant
terms have strictly positive `p`-adic valuation, contradicting the old
conclusion. Each factor must have degree at least two, their product must have
degree at least six, and `ell` must be at least two.

The product's lower Newton polygon must contain the primitive negative-slope
edge from `(ell,v(a_ell))` to `(j,0)` required by the old lemma and at least one
different negative-slope edge to its left. Your witness must also demonstrate
that at least one added left-edge condition in the corrected statement fails,
so it does not refute the repair.

Coefficient arrays are in ascending degree order and use canonical decimal
integer strings. Write a concise mathematical explanation to
`/app/evidence/answer.txt` covering the Newton polygon analysis: the old
right-edge hypotheses hold at the submitted indices, the factor constant-term
valuations contradict the old conclusion, and at least one corrected left-edge
condition fails so the witness does not refute the repair. Do not claim a
formal verification of the general Newton-polygon theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit a prime, two canonical ascending integer coefficient arrays, and indices ell,j. Each coefficient string is limited to 30 characters. The verifier derives the product, valuations, lower hull, old-hypothesis failure, and corrected-hypothesis boundary. The evidence file must contain a mathematical explanation covering the Newton polygon analysis, the old hypothesis failure, and the corrected left-edge repair boundary; unrelated or empty text does not earn evidence credit. In the solver's own words, the limitations array must disclose that Dumas's theorem or the corrected general lemma is not formally verified here.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `OLD_LEMMA_REFUTED_WITH_REPAIR_BOUNDARY`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
