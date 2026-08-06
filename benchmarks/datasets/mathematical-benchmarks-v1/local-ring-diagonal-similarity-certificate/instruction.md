# Certify diagonal similarity over a local ring

For the frozen diagonal matrices `A,B` and matrix `P` over `Z/125Z`, certify `PA=BP`, invertibility of `P`, and the resulting diagonal-entry permutation. Submit both matrix products, the determinant modulo 125, a row-to-column permutation selecting only unit entries of `P`, its sign and signed determinant term, and the six matched diagonal pairs. The verifier recomputes the full determinant expansion and every modular relation.

Bind one text explanation as `evidence/answer.txt`. The independently replayed typed certificate belongs in `submission.json`; no duplicate private serialization is required in the prose. Do not claim the general theorem is machine verified.

The explanation must affirmatively state the certified relationships: that the modular products agree, that the determinant is a unit modulo 125, and that the selected permutation matches unit entries on the diagonal. Equivalent phrasing is accepted; contradictory or unrelated text is rejected. The evidence artifact has no size cap beyond the verifier workspace, but must remain a digest-bound regular file at the declared path.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

Submit the exact modular matrix products, determinant residue, one unit determinant term, and its induced diagonal matching. The verifier recomputes the determinant expansion and every claimed ring operation. In the solver's own words, the limitations array must disclose that the certificate covers only the frozen matrix instance, not the general local-ring theorem.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission envelope requires `task_id`, `conclusion`, `result`, `claimed_assurance`, `scope`, `completeness`, `evidence`, and `limitations`.

- **Conclusion:** exactly `DIAGONAL_ENTRIES_MATCH_BY_UNIT_PERMUTATION`
- **Assurance:** scoreable values are `UNVERIFIED`, `COMPUTED` (ceiling `COMPUTED`); the submission schema accepts any of `UNVERIFIED`, `COMPUTED`, `CHECKED`, `VERIFIED` but only scoreable assurances receive credit.
- **Scope:** the exact value declared in `submission_schema.json`
- **Completeness:** `COMPLETE`.
- **Evidence:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`.
- **Evidence media types:** `text/plain`.
- **Required artifact filenames:** `evidence/answer.txt`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
