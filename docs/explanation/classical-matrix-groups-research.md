# Classical matrix groups: research disposition

**Decision: defer additional classical-group families.** Reviewed 2026-09-25
against `main` at `45eb09d0076bfb18671b4d4809e8aaff697d4104`.

This decision completes the admission research requested by
[#1884](https://github.com/morluto/jacobian/issues/1884). It does not remove
the already published bounded `GL(n,q)` and `SL(n,q)` slices, or claim that
those slices provide every classical group or every possible action.

## Evidence and current ownership

The catalog on the reviewed baseline publishes prime-field general and special
linear group constructors and their actions on nonzero vectors:

- `finite_matrix_group.general_linear.construct`
- `finite_matrix_group.special_linear.construct`
- `finite_matrix_group.general_linear.nonzero_vector_action.compute`
- `finite_matrix_group.special_linear.nonzero_vector_action.compute`

These are the narrow GL/SL slice delivered by
[#3813](https://github.com/morluto/jacobian/pull/3813). They preserve the
field, dimension, concrete matrices, exact group order, and natural action.
They do not construct symplectic, orthogonal, unitary, or semilinear groups,
and they do not identify a projective quotient with its covering matrix group.

Ready [PR #3969](https://github.com/morluto/jacobian/pull/3969) separately adds
bounded extension-field GL/SL values and their actions on projective points.
It explicitly does not claim PGL/PSL quotients or other form-preserving
families. This is active ownership, so a parallel extension-field or
projective-action implementation would duplicate it.

The benchmark search found one direct classical-matrix-group candidate:
[`elementwise-fixed-no-global-invariant` gap record](../../benchmarks/datasets/mathematical-benchmarks-v1/elementwise-fixed-no-global-invariant/analysis/gap.json).
Its gap record is historical, answer-visible, and marked `DEFER_RESEARCH`; it
asks for a determinant-one finite matrix group and a fixed-vector certificate.
This is one useful regression signal, not two independent reusable demand
families. Other nearby finite-group, projective-geometry, root-system, and
Lie-algebra tasks have contracts expressible through their respective existing
domains; a mention of a group name does not establish demand for a complete
named matrix group and its faithful natural action. No second independent
benchmark was found whose proof-critical input requires such a classical-group
value.

## Why the broader family stays deferred

The remaining candidates do not share one complete value contract. Symplectic
groups need a bound nondegenerate alternating form; orthogonal groups need
quadratic-form semantics, especially in characteristic two; unitary groups
need a specified extension field and involution. Projective groups additionally
need a canonical scalar quotient and quotient action. A list of generators is
not a complete named group without an exact completeness argument. No
independent demand currently justifies selecting one of these distinct
contracts.

## Promotion criteria

Reopen promotion when the corpus contains at least two independent reusable
families whose proof-critical input requires a still-missing full classical
matrix group and its natural action, with at least one composing into another
domain. A follow-up must name one family and its exact field/form conventions,
bind its faithful representation and any quotient explicitly, separate exact
order formulas from bounded element materialization, and provide an
independent completeness oracle. Do not count parameter variations of one
benchmark family as independent demand.

Until then, use the existing GL/SL and generic group/action values where their
contracts suffice. Keep orthogonal, symplectic, unitary, and semilinear
semantics out of a shared generic `classical_group` union.
