# Exact root--critical-point distance profiles

`polynomial.root_critical_distance_profile.compute` returns the complete
Cartesian relation between the distinct complex roots of a bounded nonconstant
univariate polynomial `p` over `QQ` and the distinct roots of `p'`. Each axis
retains its source multiplicity and a backend-independent indexed algebraic
root identity with a certified rational isolating rectangle.

Each row contains the exact nonnegative value

```text
        (a - w) * conjugate(a - w)
```

as a `RealAlgebraicValue`, together with a rational isolating interval for the
selected real root of its minimal polynomial. A coincident root and critical
point is retained as a `ZERO_DISTANCE` row with exact value zero. The profile
does not choose a nearest critical point, take a square root, or decide a
Sendov-type inequality.

Admission accounts separately for source degree and coefficient height,
distinct root and critical counts, the complete `r*c` pair expansion, and exact
distance degree before pair materialization. The result is bounded by
mathematical component and row counts, not by transport serialization size.
Backend failure or an exceeded envelope is an operational/resource
non-conclusion, never an empty family.

The current public carrier admits source degree at most eight and irreducible
root/derivative factors through degree four, keeping the exact distance carrier
within degree sixteen. It uses
maintained SymPy algebraic arithmetic privately; no `RootOf`, decimal root, or
backend session crosses the native or MCP boundary. A future normal/splitting
field value can replace the private root carrier without changing the profile's
source, multiplicity, pair-axis, or squared-distance semantics.
