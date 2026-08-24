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

The public leaf intentionally excludes arbitrary forms, spaces, bases, Sturm
bounds, Hecke/U/V operators, characters, and a global equality checker. Those
need additional canonical carriers and a pinned deterministic backend
convention.
