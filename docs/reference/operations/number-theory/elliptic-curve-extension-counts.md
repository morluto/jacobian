# Finite-field elliptic-curve extension counts

`elliptic_curve.finite_field.extension_counts.compute` returns the exact
cardinality of a nonsingular short-Weierstrass curve over each field
`F_(q^n)` for `1 <= n <= max_degree`. It obtains the base trace from the
exhaustive point count over `F_q`, then applies

```text
S_0 = 2
S_1 = t
S_n = t S_(n-1) - q S_(n-2)
#E(F_(q^n)) = q^n + 1 - S_n
```

The operation admits only base fields with `q <= 4096` and degrees at most 64.
It preflights a conservative integer-growth bound before enumerating the base
curve, and returns the base cardinality, base trace, each recurrence power sum,
and each extension cardinality. Admitted degrees let extension counts exceed
the interoperable JSON integer range, so these exact integers serialize as
canonical decimal strings. Its trace is not accepted from caller input;
the operation derives it through exhaustive exact enumeration. This keeps the
result independent of unverified Hasse-compatible trace claims.

This operation computes counts only. It does not construct extension-field
presentations or enumerate extension points.

[Number theory operations](index.md) · [Operation reference](../index.md)
