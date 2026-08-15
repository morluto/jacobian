# Elementwise fixed vectors without a global invariant

The offline input freezes a claim that swaps two quantifiers for finite linear
actions. Refute it by constructing a subgroup of `SL_3(F_q)` for one allowed
odd prime.

Submit two generators, the complete generated group in lexicographic matrix
order, and one nonzero fixed vector for every listed group element. The
verifier independently closes the generators under multiplication, checks
determinants, replays every fixed-vector equation, and computes the common
fixed-space intersection. The group must have order between 6 and 48, and its
common fixed space must be zero.

This is not a request to reproduce the public example. Alternative generators,
fields, conjugates, and fixed vectors are accepted whenever they satisfy the
contract. Explain the quantifier failure in `evidence/answer.txt`: the
explanation must cover the elementwise fixed vectors (each element fixes a
nonzero vector), the absence of a common fixed vector (no single nonzero
vector is fixed by all elements), and the quantifier-order separation that
makes the implication fail. Bind that file by SHA-256.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier independently generates the finite group, checks determinant one and every elementwise fixed vector, and computes the common fixed-space intersection. The evidence file remains qualitative prose and must explain the elementwise fixed vectors, absence of a common fixed vector, and quantifier-order separation; unrelated or empty text does not earn evidence credit. The limitations array records the exact obligation IDs `claim:finite-action-counterexample` and `limitation:no-general-classification-theorem` in that order.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result` and the declared `witness`.

- **Witness:** 1-1 item(s); allowed path(s): `evidence/answer.txt`; digest must match `^sha256:[0-9a-f]{64}$`; media type(s): `text/plain`.
<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
