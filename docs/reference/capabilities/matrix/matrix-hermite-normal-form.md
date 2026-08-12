# Integer matrix Hermite normal form

`matrix.normal_form.hermite.materialize` computes and retains one exact row
Hermite normal form for a finite integer matrix. Its independent checker is
`matrix.normal_form.hermite.verify`; computation and verification are separate
trust boundaries.

This row-Hermite outcome is distinct from the full two-sided relation
documented in
[Certified Smith normal form and integral homology](../../scenarios/certified-smith-integral-homology.md).
Neither normal form is presented as a replacement or preferred strategy for
the other.

## Input contract

The producer accepts one [`IntegerMatrix`](index.md#shared-matrix-values)
from `jacobian.contracts.matrices`: a nonempty rectangular matrix over `ZZ`
with 1 to 32 rows and 1 to 32 columns. The shared `IntegerMatrix` model
permits up to 32,768 canonical digits per entry; the HNF request model
tightens this to 256 decimal digits per entry via its own
`require_matrix_scalar_digits` validator.

The request includes a wall-clock budget from 1 to 60 seconds. The default is
10 seconds.

## Computed evidence

The base package pins Python-FLINT 0.9.0 and its linked FLINT 3.6.0
library for this profile. Its HNF operation calls
`fmpz_mat.hnf(transform=True)` in an isolated process and returns two exact
integer matrices:

- `H`, the proposed row Hermite normal form; and
- `U`, the proposed left transformation satisfying `H = U A`.

The [Python-FLINT matrix documentation][python-flint-hnf] exposes this exact
transformation relation. The materialized operation stores the bounded input
and complete `H`/`U` certificate as explicit artifacts, along with the
provider runtime and resource budget. A successful provider call has
`COMPUTED` assurance and `conclusion: UNKNOWN`; it does not verify its own
output.

Timeout, process failure, an invalid worker response, excessive output, or a
runtime identity change produces no normal-form artifact and no mathematical
conclusion.

## Row-HNF convention

The candidate uses FLINT's row convention:

1. every zero row follows every nonzero row;
2. the first nonzero entry of each nonzero row is positive;
3. pivot columns increase strictly down the rows; and
4. every entry above a pivot lies in the half-open interval `[0, pivot)`.

These conditions match FLINT 3.6.0's
[`fmpz_mat_is_in_hnf` implementation][flint-is-in-hnf]. Entries to the left of
a pivot are zero because the pivot is the row's first nonzero entry.

## Independent verification

With bundled references enabled, `matrix.normal_form.hermite.verify` runs an
operator-authorized standard-library checker in a clean process. The checker
accepts only when all of the following hold:

1. artifact schemas, semantics, payload digests, exact source bindings, and
   parent lineage agree;
2. `H` and `U` have the dimensions bound to `A`;
3. exact integer multiplication gives `H = U A`;
4. an independent fraction-free Bareiss determinant gives `det(U) = 1` or
   `det(U) = -1`; and
5. every row-HNF convention condition above holds.

Only acceptance creates a verification record and returns `VERIFIED`.
Rejection, timeout, cancellation, malformed evidence, or checker failure
returns `UNKNOWN`. Rejection does not prove any opposite normal-form claim.

## Artifact bindings

The source matrix and HNF candidate are immutable artifacts under one
versioned integer-matrix semantics descriptor. The `H` and `U` matrices in
the candidate are `IntegerMatrix` values—the same shared contract type used
by the request and by other integer-matrix operations. The candidate binds:

- the source artifact URI, object digest, and payload digest;
- the exact source dimensions;
- the complete `H` and `U` matrices;
- the pinned producer runtime identity and operation profile; and
- the enforced resource budget.

The verification witness separately binds the exact source, candidate,
semantics, and checker identity. Replacing any source entry, transformation,
normal-form entry, digest, binding, or lineage edge invalidates replay.

[python-flint-hnf]: https://python-flint.readthedocs.io/en/latest/fmpz_mat.html#flint.fmpz_mat.hnf
[flint-is-in-hnf]: https://github.com/flintlib/flint/blob/v3.6.0/src/fmpz_mat/is_in_hnf.c
