# Certify a reciprocal-polynomial family member

The nonconstant real-polynomial solutions of
`1/P(z)+1/P(1/z)=z+1/z` are claimed to be

`P_m(z) = sum_{j=0}^{m-1} (-1)^j z^(2j+1)` for positive integers `m`.

Choose any `m` from 6 through 20 and submit an exact sparse coefficient list
for `P_m`, the coefficients of `Q=P_m/z`, the reversed polynomial
`S=z^(2m-2)Q(1/z)`, the constant relating `S` and `Q`, and the quotient in
`1-(-z^2)^m=(1+z^2)Q`. The verifier will independently reconstruct the
cleared Laurent identity; numerical sampling is not accepted.

Write `submission.json` and digest-bind
`evidence/classification-certificate.json`, which must copy `result` and
`limitations` exactly. Claim at most `COMPUTED`.
