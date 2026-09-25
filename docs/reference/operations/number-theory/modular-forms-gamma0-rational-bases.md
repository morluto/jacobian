# Rational Gamma0 modular-form bases

`modular_form.space.basis_q_expansions.compute` also returns exact
q-expansion bases for bounded rational, trivial-character spaces
`M_k(Gamma0(N))` and `S_k(Gamma0(N))` beyond the formula-based levels already
documented in [level-one bases and coordinates](modular-forms-level-one-bases-coordinates.md).
The initial backend-backed slice admits `N <= 10,000`, `k <= 120`, dimension at
most 32, and at most 128 q-coefficients. A request must include the full
Sturm-determining prefix; shorter prefixes cannot establish basis completeness.

The implementation uses PARI's exact modular-form-space routines behind a
typed adapter. It checks PARI's dimension against Jacobian's independent exact
Gamma0 dimension formula, then removes backend ordering and scaling by
canonical exact row reduction through the Sturm prefix. The result uses a
stable Jacobian basis identifier and the existing modular-form basis and
coordinate carriers. Coordinates can therefore be expanded in the same
declared space through the existing coordinate q-expansion operation.

PARI's reference documents `mfinit` for full `M_k(Gamma0(N), chi)` spaces and
cuspidal `S_k` spaces, `mfbasis` for the space basis, and `mfcoefs` for the
matrix of basis q-expansions. This general Gamma0 adapter remains restricted
to trivial character and rational coefficients. Separately, Jacobian has a
bounded character-valued basis and coordinate slice for the even order-6
characters modulo 13 in `S_2(Gamma0(13), chi)` over `Q(zeta_6)`; see
[level-one bases and coordinates](modular-forms-level-one-bases-coordinates.md)
for its exact q-prefix and Sturm-bound contract. Other
character spaces and general field-valued Gamma0 bases remain unsupported.

For this represented character slice,
`modular_form.character_hecke_matrix.compute` returns the exact 1-by-1 matrix
of `T_n` in the canonical character basis for `1 <= n <= 32` with `gcd(n,
13) = 1`. Its entry is bound to the exact source space, basis identifier, and
`Q(zeta_6)` coefficient field. The operation admits the required finite source
prefix and coefficient heights before invoking PARI, then checks the matrix
action through the Sturm-determining prefix. It makes no claim about Hecke
matrices on other character spaces.
[PARI modular-forms reference](https://pari.math.u-bordeaux.fr/dochtml/ref-stable/Modular_forms.html)

Before backend work, Jacobian admits level, weight, dimension, Sturm
precision, aggregate work, rational elimination growth, and serialized output.
The PARI call runs in a request-scoped resource-limited worker so it can be
terminated under the request deadline. Backend objects never cross the
adapter; only bounded exact rational coefficients return to the canonical
Jacobian representation.

The accepted rational envelope is an execution bound, not a claim that every
Gamma0(N) space in that range will fit. Requests exceeding the dimension,
Sturm precision, work, coefficient-growth, or output limits are rejected
before basis construction. Wider character support, field-valued coefficients,
and operators on arbitrary Gamma0(N) spaces require additional parent-bound
representations and are not implied by these bases.

[Number-theory operations](index.md) · [Modular-form basis operations](modular-forms-level-one-bases-coordinates.md)

## Nested-space coordinate transport

`modular_form.coordinates.transport.compute` expresses an exact coordinate-defined
form in a target space when both are over `QQ`, have trivial character and the
same weight, and the source level divides the target level. The full space maps
to the full space; a cuspidal form maps to either the cusp space or full space.
It computes the source expansion through the target Sturm-determining precision,
then solves uniquely in the target's canonical basis. The returned
`ModularFormCoordinates` is bound to the exact target space and basis.

Both basis plans, their combined work, coordinate growth and result size are
admitted before basis materialization, including before PARI worker calls.
Current limits remain those of the bases: level at most 10,000, weight at most
120, dimension at most 32 and Sturm precision at most 128. Cyclotomic basis
transport and nontrivial-character transport are unsupported; a representable
space value does not imply its basis or coordinate carrier supports that field.
