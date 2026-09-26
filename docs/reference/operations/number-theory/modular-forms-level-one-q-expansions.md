# Level-one named modular-form q-expansions

`modular_form.level_one.named_q_expansion.compute` constructs a complete exact
prefix of one closed normalized family in `QQ[[q]]`: `E4`, `E6`, or Ramanujan
`DELTA`. The returned value retains the level-one `SL2Z` parent, weight,
holomorphic/cuspidal space kind, normalization, and a canonical
`TruncatedSeries` through the requested precision.

For `P`, the series has exactly the coefficients of `q^0` through `q^(P-1)`;
it does not claim anything about later coefficients. `E4` and `E6` use their
exact divisor-sum definitions, and `DELTA` is constructed as
`(E4^3 - E6^2) / 1728` through the existing exact truncated-series operations.

This constructor is limited to its three named level-one forms. Separate
operations compute supported space dimensions and Sturm bounds, and construct
level-one bases and exact rational coordinate forms. Coordinate values retain
their exact space and basis identity for caller composition.
Exact basis and coordinate values are available for holomorphic `M_k(Gamma0(2))` with trivial character over `QQ`,
using the graded-ring generators `A2=2E2(2tau)-E2(tau)` and `E4`. The
Gamma0(2) slice supports weights through 120 and exact q-prefixes through
precision 128, subject to result-growth admission. It does not yet provide
Gamma0(2) cusp bases or Hecke/U/V actions. Exact `T_n` actions and level-raised
`U_p`/`V_p` images remain available for level-one coordinate forms. Arbitrary
characters, other higher levels, and arbitrary basis arithmetic remain
unsupported. Generic finite-prefix transforms that accepted a caller-declared
modular space remain unsupported: a q-prefix cannot establish that its source
belongs to the declared space. The separate formal `U_p` and `V_p` prefix maps
return only `TruncatedSeries` values and make no modularity claim. The supported
modular operator actions accept canonical basis coordinates, whose q-expansions
and space membership are defined by the basis construction. Exact character-bound dimensions, Sturm bound, and bases
also support M_1 and M_3 on Gamma0(4) with chi_-4; the two-dimensional
weight-three basis is documented separately. These operations return basis
coordinates, Sturm integers, and finite q-prefixes; callers can compose those
values with ordinary exact coefficient comparisons. Cross-space equality for
the order-six `S_2` character slice at levels 13, 26, and 39 is supported only
through the explicit bounded inflation and common-target comparison documented
in [Rational Gamma0
modular-form bases](modular-forms-gamma0-rational-bases.md).
