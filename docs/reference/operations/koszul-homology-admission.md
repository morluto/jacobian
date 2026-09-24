# Koszul homology exact-rank admission

The finite-module Koszul homology operation returns the source complex along
with three integer profiles. Its exact linear algebra uses rational
Gauss-Jordan rank, and it validates the source action and `d^2=0` before the
rank calculations. All mandatory phases run only after the homology admission
checks below.

## Coefficient and output bounds

Every numerator and denominator retained in the returned complex must have at
most 128 decimal digits. This applies to the algebra, the nested module
algebra, module actions, sequence, and differential matrices. The validator
rejects larger canonical values before replaying actions, multiplying
differentials, or entering rational elimination.

The output estimate counts decimal digits for every retained rational, the
sparse matrix coordinates, basis labels, and fixed structure overhead. It
rejects a result estimate over 8 MiB. The source chain basis is bounded by 256,
so homology dimensions and cycle/boundary dimensions each fit in `0..256`, and
at most seven degrees are returned.

## Exact-rank growth bound

For one sparse differential with `m` rows and `n` columns, let `t=min(m,n)`.
For row `i`, let `D_i` be the product of denominators of its nonzero entries;
the implementation estimates `log2(D_i)` by summing the denominator bit
lengths. Clearing those row denominators gives an integer matrix `B`. If `E`
is the maximum bit bound on an entry of `B`, Hadamard's inequality gives the
following upper bound for every `s` by `s` minor of `B`:

```text
s * (E + ceil(log2(s))) bits.
```

At elimination step `k`, each reduced Gauss-Jordan entry is a Schur complement,
which equals a ratio of minors of sizes `k+1` and `k`. Clearing the row scales
adds at most the largest `log2(D_i)` to the denominator bound. The operation
uses the resulting `P` as a bound for numerator and denominator bit lengths of
every reduced Fraction through all `t` pivots.

Multiplying two values bounded by `P` needs at most `2P` component bits. A
subtraction can cross-multiply those values and add the resulting numerators;
`4P+2` therefore bounds each transient component used by a pivot update. The
admission counts the dense pivot scan, row normalizations, two rational
operations for every possible row update, and consecutive-differential
products. Each differential rank is computed once and reused as the outgoing
and incoming rank in adjacent degrees. The estimate weights rank operations by
the square of the transient bit bound and square-check operations by the
corresponding bound for their partial sums. It rejects work estimates above
`2^40` units. The estimate is
deliberately conservative; it may reject some inputs that would reduce cheaply.

These limits bound exact intermediate growth for the current rational kernel;
they are admission limits rather than timeouts. Resource refusal does not
establish any homology or exactness conclusion.
