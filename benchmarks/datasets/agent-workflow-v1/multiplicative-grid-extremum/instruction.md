# Certify the maximum number of multiplicative neighbors

The frozen source asks for the maximum possible number of good pairs among 100
distinct positive integers. A pair is good when its larger member is exactly
2 or 3 times its smaller member.

Produce `/app/submission.json` matching `/app/submission_schema.json` and one
evidence file at `/app/evidence/answer.txt`.
The evidence file must be no larger than 16 MiB (16,777,216 bytes).

Your result must contain:

- exactly 100 distinct positive integers within the declared bounds;
- the complete set of exactly 180 good index pairs `[i,j]`, with `i < j`;
- for every number at index `i`, its unique factorization
  `core * 2^two_exponent * 3^three_exponent`, where `gcd(core,6)=1`;
- the number of multiplicative components and the total numbers of nonempty
  horizontal rows and vertical columns in their exponent lattices;
- the witness projection cost `rows + columns`;
- the universal projection cost 20 and resulting edge bound 180.

For the upper bound, use the following elementary projection argument. Within
each fixed core, horizontal edges are at most the number of vertices minus the
number of occupied rows; vertical edges satisfy the analogous column bound.
A component occupying `m` exponent pairs needs row and column counts `r,c`
with `m <= r*c`. Minimize the total `r+c` over every partition of 100 vertices.
The verifier independently performs this finite minimization.

The evidence file must contain exactly one line beginning with `RESULT_JSON:`
followed by the exact compact JSON encoding of the submitted `result`. The
verifier binds that marker to the submitted result; additional text is not
scored. For example:

```json
RESULT_JSON: {}
```

Replace the empty object with the exact `result` object in the submission and
bind the evidence file by its `sha256:` digest. The verifier parses the marker
as JSON and requires semantic equality with the submitted result.

Use `claimed_assurance: "COMPUTED"`. `VERIFIED` is unsupported because the
trusted projection lemma is not replayed by a proof assistant. Completeness is
`COMPLETE`; the exact scope and limitation strings are fixed by the schema.
