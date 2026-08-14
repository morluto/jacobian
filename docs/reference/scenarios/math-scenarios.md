# Mathematical scenario catalog

[Documentation home](../../index.md)

- Purpose: reusable mathematical cases for documentation, tests, and
  benchmarks
- Status: reference inputs and expected mathematical outcomes, not runtime
  workflows

## Scenario record

Each scenario identifies:

- exact mathematical input and parent/domain;
- expected value or bounded-search outcome;
- any domain-owned bounds, coverage status, or witness;
- applicable operation or native mathematical function; and
- adversarial mutations that must be rejected at the typed request boundary.

A scenario does not prescribe tool order. Agents may use a known operation
directly, search first, compose several values, or stop with the result it has.

## Exact arithmetic

### INT-GCD-001 — Greatest common divisor

Input: `84` and `30` over the integers.

Expected value: `6`.

This is a scalar ordinary-operation case and is useful as an MCP/CLI parity
smoke.

### INT-FACTOR-001 — Composite integer

Input: `91`.

Expected factorization: `7 × 13`.

The returned factors are a mathematical value. A failed factor search is not
evidence that the input is prime.

## Matrices

### MATRIX-DET-001 — Exact rational determinant

Input over `QQ`:

```text
[[1,  0, 1],
 [2, -1, 3],
 [4,  3, 2]]
```

Expected determinant: `-1`.

`matrix.determinant.compute` returns the exact value inline. Mutate a rational
denominator to zero or change a matrix entry to exercise request validation and
the resulting value.

### MATRIX-RANK-001 — Exact rank and pivots

Input over `QQ`:

```text
[[1, 2, 3],
 [2, 4, 6],
 [0, 1, 1]]
```

Expected rank: `2`. A consumer receives the typed matrix value it requires; a
same-shaped value with different entries is not a substitute.

## Polynomials

### POLY-RESULTANT-001 — Exact resultant

Use two exact `sympy.Poly` values with the same generator and explicit `QQ`
domain. The public Python function accepts those canonical semantic inputs;
wire conversion to sparse polynomial contracts remains outside the function.

Test generator mismatch, coefficient-domain mismatch, output-size refusal, and
agreement between the Python API and catalog operation.

### POLY-SEPARATE-001 — Ideal membership or separator

Given an exact polynomial and generators, return a typed membership
representation or a typed separating outcome. A solver status alone is not a
broader mathematical claim.

## Graphs

### GRAPH-TRIANGLE-001 — Triangle count

Input: one undirected simple triangle with stable node labels.

Expected count: `1`.

The public Python function and installed graph invariant operation share one
implementation. Reject directed graphs and mutable/backend values whose labels
or canonical ordering cannot satisfy the public identity contract.

### GRAPH-DIAMETER-001 — Connected path

Input: the path graph on four vertices.

Expected diameter: `3`.

A disconnected graph is outside the operation's domain unless the exact
operation contract defines a component-wise result. It is not silently assigned
an infinite or sentinel diameter.

## Prime-field linear algebra

### PRIME-RANK-001 — Rank over `F₂`

Input entries are immutable exact residues and the prime `2` is explicit.
Compare `jacobian.math.prime_field_linear_algebra.rank` with
`python-flint.nmod_mat(..., 2)` across rectangular, singular, and full-rank
cases. The module imports no topology or runtime code.

### PRIME-NULLSPACE-001 — Basis round trip

Compute RREF, pivot columns, nullspace, column basis, and quotient basis from
the same explicit prime-field value. Check that returned vectors satisfy the
defining equations and preserve canonical basis ordering.

## Finite-field composition

### Direction rank ledger

Construct the exact presentation

```text
F₂[a] / (a³ + a + 1)
```

binding characteristic, irreducible polynomial, generator, ordered power
basis, encoding version, and presentation digest. Enumerate the nine normalized
projective directions, explicitly restrict scalars, construct

```text
B ↦ Bᵀb : F₂⁴ → F₂⁶
```

and attach every direction to its rank before aggregating the orbit
distribution.

Reject a differently presented isomorphic field, wrong axes or bases, a
reinterpreted `2×2` rank over `F₈`, missing directions, and substituted ranks.

### Finite polynomial map fibers

Reuse the exact field presentation and element encoding from the direction
ledger scenario. Enumerate a bounded polynomial map table whose domain and
codomain are exact. The fiber partition covers the complete field exactly once.
Collision and permutation certificates bind the map and enumeration scope.

Reject a second field representation, incomplete table, duplicate or missing
domain element, wrong codomain parent, or forged fiber.

## Scenario use

Public examples should remain small enough to understand and run. Larger or
held-out variants may live in evaluation datasets, which are separate from the
server's operation contract.
