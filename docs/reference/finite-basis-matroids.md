# Finite basis matroids

`jacobian.math.combinatorics.matroids.FiniteBasisMatroid` is a canonical
finite matroid value represented by its complete family of bases. It supports
matroids that are not known to have a representation over any particular
field. This fills the value-layer prerequisite for delta-matroid lower and
upper matroids described in [#1954](https://github.com/morluto/jacobian/issues/1954).

The retained value is:

```text
ground: unique ordered labels
bases: nonempty, complete family of sorted ground-index tuples
```

The rows must have a common cardinality. For each ordered pair of bases
`B1, B2` and each `x` in `B1 - B2`, some `y` in `B2 - B1` must make
`(B1 - {x}) ∪ {y}` a basis. This ordinary basis-exchange axiom characterizes
finite matroids in the basis presentation; it does not require the stronger
symmetric or bijective exchange property. See the
[Mathlib basis definition](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Combinatorics/Matroid/Basic.html)
for a formalized basis-first matroid API.

The ground axis remains explicit when some elements are loops. The empty
ground matroid and a rank-zero matroid on a nonempty ground both have the one
basis `()`. The rank is the common basis cardinality.

Construction checks the complete basis axiom after admitting the input under
all of these limits:

| Quantity | Bound |
| --- | ---: |
| Ground elements | 64 |
| Basis rows | 4,096 |
| Total basis memberships | 65,536 |
| UTF-8 bytes per ground label | 1,024 |
| UTF-8 bytes across ground labels | 16,384 |
| Worst-case exchange candidate checks | 2,000,000 |

The exchange estimate is `m² min(r, n-r)²`, where `m` is the number of bases,
`r` is their common cardinality, and `n` is the ground size. It bounds every
ordered base pair and each candidate one-element exchange before the axiom
scan begins. Requests that exceed an envelope are rejected without a
mathematical conclusion about their basis family.

This carrier is separate from `LinearMatroid`: it asserts ordinary matroid
basis exchange, not linear representability or a field, matrix, or backend
presentation. It publishes no operation by itself; callers compose the value
with operations that consume basis families.
