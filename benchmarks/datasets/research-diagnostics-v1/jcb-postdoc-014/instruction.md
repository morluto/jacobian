# Nine-line counterexample to combinatorial determination of relation degree

The frozen input gives two labelled arrangements as integer coefficient triples
for their nine linear factors over Q[x,y,z]. For a homogeneous degree-nine
polynomial h, define mdr(h) as the least q for which nonzero homogeneous
degree-q polynomials A,B,C satisfy A h_x + B h_y + C h_z = 0.

Verify that the two arrangements have exactly the listed common non-double
flats, prove mdr(f)=4 and mdr(g)=5, and conclude that the stated generalized
Terao-type conjecture is false. Lower-degree injectivity and the first nonzero
relations must both be established; failed witness search is not injectivity.

Write the flats and one nonzero integer-coefficient relation at each first
degree to `evidence/syzygy-certificate.json` using the exact agent-visible
`certificate_schema.json`. Sparse terms must be in descending monomial order,
with unique exponent triples summing to the declared degree. Bind the file in
`submission.json` with its SHA-256 digest and exact submission schema.

This is a public answer-visible diagnostic. Claim `COMPUTED` only. The
benchmark verifier independently reconstructs both arrangements, proves the
lower-degree ranks, and checks both polynomial identities; it is not product
verification authority.
