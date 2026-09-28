# Proper-hypergeometric shift-operator action

`ore.proper_hypergeometric.apply_shift_operator.compute` applies a bounded
left-coefficient shift Ore operator

```text
P = sum_i p_i(n) S^i,       (S f)(n) = f(n+1)
```

to a canonical proper-hypergeometric term `T(n,k)`. It returns the exact
rational function `R(n,k)` in `QQ(n,k)` satisfying

```text
P T(n,k) = R(n,k) T(n,k)
```

on the common generic locus where the term and shifted terms are defined and
nonzero. The operation composes the term's exact `n` shift quotient from
`ore.proper_hypergeometric.shift_quotients.compute` and preserves both source
values in the result. The result does not define quotients at zeros or
reciprocal-factorial support boundaries and does not claim a finite-sum
identity or boundary contribution.

The operator uses the shared shift-operator carrier's order bound. Before
rational-expression normalization, the operation bounds the common numerator
and denominator term counts, total degree, and coefficient digits of the
summed relative multiplier. The exact result is limited to 256 terms per
polynomial, total degree 128, and 128-digit coefficients. Requests whose
worst-case cross-multiplied expansion exceeds these limits are rejected before
expansion; lower-order operators remain accepted whenever their derived
expansion fits.

The exact symbolic normalization runs in a killable worker under the request's
remaining deadline. A timeout or cancellation means no multiplier was computed
and establishes no mathematical conclusion.

This operation supplies the operator-action primitive for constructing and
replaying creative-telescoping identities; it does not search for a
telescoper, interpret summation bounds, or assign values at support boundaries.
