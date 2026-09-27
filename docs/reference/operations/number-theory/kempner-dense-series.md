# Dense Kempner reciprocal-series enclosures

The private fixed-point kernel used by `enclose_kempner_series` encloses the infinite
reciprocal series for a proper base-b digit family without constructing
the potentially enormous common denominator of its finite partial sum.

For a requested cutoff D and decimal precision q, each accepted positive
integer n with at most D digits contributes
`floor(10^q/n) / 10^q` to the lower endpoint and
`ceil(10^q/n) / 10^q` to the upper endpoint. The operation adds the exact
geometric tail bound

`r * (s/b)^D / (1 - s/b)`,

where s is the number of allowed digits and r is the number of allowed
nonzero digits. Thus the returned rational interval contains the entire
infinite series. Its width is at most the finite term count times `10^-q`
plus the stated tail bound.

Admission bounds the finite family at 600,000 numerals, prefix traversal at
2,000,000 visited nodes, the explicit traversal stack at 1,000,000 entries,
and preflights the result rational height before enumerating any numeral.
The exact-partial-sum
operation remains available for smaller families where callers need that
partial sum as an exact value. The dense operation returns an exact interval,
not a density or a convergence certificate beyond the displayed tail bound.
