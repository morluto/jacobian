# Classical matrix group admission evidence

Reviewed 2026-09-25 for issue [#1884](https://github.com/morluto/jacobian/issues/1884).

## Decision

Defer additional classical-group families. The catalog and current issue corpus
do not show the two independent direct-demand families required by #1884 to
promote another family. Keep #1884 open and deferred; do not create speculative
family issues until that evidence exists.

## Existing operation ownership

PR [#3813](https://github.com/morluto/jacobian/pull/3813) supplies the bounded
prime-field `GL(n,q)` and `SL(n,q)` constructors and nonzero-vector actions.
Open READY PR [#3969](https://github.com/morluto/jacobian/pull/3969) supplies
extension-field `GL/SL` and projective-point actions. Neither slice claims
`PGL/PSL` quotient semantics or symplectic, orthogonal, unitary, or semilinear
groups.

## Corpus evidence

The only direct classical-matrix-group candidate found in the current
mathematical benchmark corpus is the historical
[`elementwise-fixed-no-global-invariant` gap record](../datasets/mathematical-benchmarks-v1/elementwise-fixed-no-global-invariant/analysis/gap.json).
It requires an exact determinant-one matrix group and a fixed-vector
certificate. This is one candidate family, not two independent reusable
demands.

Nearby finite-group, finite-projective-geometry, root-system, and Lie-algebra
tasks were checked by their proof-critical inputs and available operation
contracts. Their current requirements can be expressed by the respective
existing owners; mentions of groups or symplectic data do not require a full
named classical matrix group with its natural action. No second independent
benchmark family with that requirement was found.

## Reconsideration criteria

Reopen promotion when at least two independent reusable families require a
still-missing full classical matrix group and its natural action, with at
least one composing into another domain. A proposal must identify one group
family, exact field/form conventions, representation and quotient semantics,
separate order computation from bounded element materialization, and provide
an independent completeness oracle. Parameter variations of one benchmark
construction do not count as separate demand.
