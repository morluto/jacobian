# Classify the nonnegative image of a cubic form

Determine exactly which nonnegative integers occur as
`A^3+B^3+C^3-3ABC` for nonnegative integers `A,B,C`.

Submit a complete certificate containing: the linear-times-quadratic
factorization; the full modulo-9 image obtained from all residue triples; the
excluded residues; and affine one-parameter constructions covering every
remaining residue class, including zero and their valid parameter domains.
Represent each affine formula `u*k+v` as `[u,v]`.

Use the factorization fields `linear: "A+B+C"` and
`quadratic: "A^2+B^2+C^2-AB-AC-BC"`.

A list of residue classes or a few numerical witnesses is insufficient.  The
verifier independently expands every family symbolically, checks
nonnegativity, and proves residue-class coverage.  Put a concise derivation in
`/app/evidence/answer.txt`, include a `RESULT_JSON:` line containing the
submitted result as JSON, and bind that file with its SHA-256 digest.  Do not
claim `VERIFIED`.
