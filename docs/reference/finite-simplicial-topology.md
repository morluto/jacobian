# Finite simplicial topology

[Documentation home](../index.md)

- Status: Current implementation reference; contracts are experimental
- Domain: `topology`
- Producer backend: Jacobian standard-library exact finite and modular arithmetic
- Checker backend: isolated standard-library replay with no producer imports

The topology bundle exposes three atomic outcomes over bounded finite abstract
simplicial complexes:

- `topology.simplicial_complex.materialize`
- `topology.simplicial_complex.chain_complex.compute`
- `topology.simplicial_homology.compute`

Each producer remains `COMPUTED`. An operator-authorized
`topology.result.verify` replay is the only path in this family to `VERIFIED`.

## Canonical finite complexes

Materialization accepts an explicit vertex list and maximal facets. Vertex
labels use a bounded ASCII identifier syntax. Caller order does not orient a
simplex: lexicographic vertex order is the sole orientation convention.

The request is rejected before computation or artifact writes when:

- a vertex or facet is repeated;
- a facet repeats a vertex or mentions an undeclared vertex;
- one supplied facet is a proper face of another;
- a declared vertex appears in no facet; or
- the dimension or complete face closure exceeds its bound.

An isolated vertex is represented by a singleton maximal facet. The empty
simplex is part of the mathematical convention but is not stored among the
non-empty faces.

The result contains canonical maximal simplices, every non-empty face grouped
by dimension, the dimension, f-vector, closure size, and a content digest over
the complete canonical complex. The generic operation envelope stores the
validated request and result as content-addressed artifacts and exposes their
relationship. A downstream topology request consumes the canonical
`result.complex` object, so it cannot silently reinterpret a raw facet list.

## Chain complexes

`topology.simplicial_complex.chain_complex.compute` accepts the canonical
complex and either:

- `INTEGER`, with no prime; or
- `PRIME_FIELD`, with a prime between 2 and 251.

For every dimension it returns the lexicographically ordered simplex basis and
the complete sparse matrix of the oriented boundary

\[
\partial_k[v_0,\ldots,v_k]
=\sum_{i=0}^{k}(-1)^i[v_0,\ldots,\widehat{v_i},\ldots,v_k].
\]

Prime-field coefficients are reduced to the canonical range
`0 <= coefficient < p`; sparse matrices omit zero entries. A ledger records
the independently recomputable claim that every adjacent boundary product is
zero.

The `REDUCED` convention exposes the augmentation \(C_0\to\mathbf F_p\) as a
separate matrix. It does not disguise the augmentation as a stored
negative-dimensional simplex. `UNREDUCED` omits it.

## Homology over a prime field

`topology.simplicial_homology.compute` is limited to one explicitly declared
bounded prime field. For every dimension it preserves:

- the outgoing boundary rank and kernel dimension;
- a complete cycle basis;
- the incoming boundary rank and a basis for its image;
- the Betti number;
- representatives for a basis of cycles modulo boundaries; and
- the rank showing that boundaries plus the quotient representatives span the
  complete cycle space.

The operation supports explicit `REDUCED` and `UNREDUCED` conventions. Its
dimension range always covers every chain group of the supplied complex.

## Bounds

Version 1 uses these fail-closed limits:

| Quantity | Limit |
| --- | ---: |
| Declared vertices | 64 |
| Maximal facets | 128 |
| Dimension | 7 |
| Complete non-empty face closure | 2,048 |
| One chain group used for linear algebra | 512 |
| One dense boundary shape | 131,072 cells |
| Coefficient prime | 251 |

A complex can be materialized when it fits the closure bound but still be
rejected by chain or homology computation when a linear-algebra bound is
exceeded. That rejection occurs during complete request validation, before
operation artifacts are written.

## Public reference cases

The regression suite includes:

| Space | f-vector | Unreduced Betti numbers |
| --- | --- | --- |
| One point | `(1)` | `(1)` |
| Three discrete points | `(3)` | `(3)` |
| One filled edge | `(2, 1)` | `(1, 0)` |
| Triangle boundary \(S^1\) | `(3, 3)` | `(1, 1)` |
| Filled triangle | `(3, 3, 1)` | `(1, 0, 0)` |
| Tetrahedron boundary \(S^2\) | `(4, 6, 4)` | `(1, 0, 1)` |
| Filled tetrahedron | `(4, 6, 4, 1)` | `(1, 0, 0, 0)` |
| Cone on a triangle boundary | `(4, 6, 3)` | `(1, 0, 0)` |
| Suspension of a triangle boundary | `(5, 9, 6)` | `(1, 0, 1)` |
| Seven-vertex torus | `(7, 21, 14)` | `(1, 2, 1)` over \(F_2\) and \(F_3\) |
| Six-vertex real projective plane | `(6, 15, 10)` | `(1, 1, 1)` over \(F_2\); `(1, 0, 0)` over \(F_3\) |

The same suite covers disjoint unions, reduced \(H_0\), vertex relabeling,
caller simplex order, boundary signs, and forged cycle evidence.

## Independent verification

The clean-process checker reconstructs the complete face closure and every
oriented boundary from passive JSON. For homology it separately:

1. checks every adjacent boundary product;
2. recomputes modular ranks;
3. checks that the reported cycle basis is an independent spanning set for
   the kernel;
4. checks that the reported boundary basis equals the incoming image;
5. checks that every quotient representative is a cycle; and
6. checks independence and spanning modulo boundaries.

Replay binds the exact input artifact, result artifact, topology semantics,
candidate digest, witness format, and checker identity. A malformed,
interrupted, unsupported, or false replay returns no opposite mathematical
conclusion and creates no verification record.

Integral homology, torsion generators, persistent homology, and
low-dimensional manifold recognition remain outside this contract. Integral
generators require certified Smith transformations; persistence requires a
separate exact-filtration and GUDHI provider gate.
