# Discover and certify a reciprocal-polynomial solution

Find a nonconstant integer-coefficient polynomial `P` of degree between 11 and
39 satisfying `1/P(z)+1/P(1/z)=z+1/z`. Discover the relevant family index `m`
between 6 and 20, then submit an exact sparse coefficient list for `P`, the
coefficients of `Q=P/z`, the reversed polynomial `S=z^deg(Q) Q(1/z)`, and the
constant relating `S` and `Q`. The verifier will independently recognize the
family member, check the geometric divisibility identity, and reconstruct the
cleared Laurent identity. Numerical sampling is not accepted.

Write `submission.json` and digest-bind
`evidence/classification-certificate.json`, which must copy `result` and
`limitations` exactly. Claim at most `COMPUTED`.
