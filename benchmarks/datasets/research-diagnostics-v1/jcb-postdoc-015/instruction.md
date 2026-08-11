# Erdős Problem 707: finite exact evidence and a public universal result

Let `A = {1,2,4,8,13}`. An integer set is Sidon when its ordered differences
`a-b`, for distinct elements, are all different. A `k`-element perfect
difference set lives modulo `k(k-1)+1` and represents every nonzero residue
exactly once by its ordered differences.

The public answer-visible result says that `A` is Sidon but is not contained in
any finite perfect difference set. Reproduce the following independently
checkable finite core:

- all 20 ordered integer differences of `A`;
- complete fixed-order extension decisions for `k=5,6,7`, using direct
  containment of the reduced residues of `A` modulo `k(k-1)+1`;
- the exact number of candidate supersets examined for each fixed order.

Write that evidence to `evidence/finite-core.json` and bind its SHA-256 digest
in `submission.json`. Follow `submission_schema.json` for the submission and
the separate agent-visible `/app/evidence_schema.json` for the complete
`evidence/finite-core.json` body. The fixed-order checks must cover every
candidate, not a sample. The public universal result may be reported, but this task does
not supply its projective-geometric or formal obstruction certificate: set
`universal_obstruction_replayed` to `false` and do not claim that the three
finite searches prove the universal statement.

The maximum allowed assurance is `COMPUTED`. Timeout, error, an incomplete
enumeration, or failure to find an extension is not a negative conclusion.
