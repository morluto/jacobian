# Certified Smith normal form and integral homology

[Documentation home](../../index.md)

- Status: Experimental contracts
- Matrix producer backend: SymPy `smith_normal_decomp` over `ZZ`; canonical
  diagonal and invariant factors, noncanonical unimodular transformations
- Topology producer backend: exact integer chain arithmetic over canonical
  simplex bases; integral homology derives cycle bases and generator
  coordinates from the certified Smith transformations
- Checker backend: isolated standard-library replay with no producer or public
  contract imports

This family adds two atomic mathematical outcomes:

- `matrix.normal_form.smith.certified.compute` returns a canonical Smith
  diagonal and both complete basis changes for one integer matrix; and
- `topology.simplicial_homology.integral.compute` returns the free and torsion
  decomposition, concrete generators, bounding chains, and the transformation
  certificates needed to inspect every integral homology group of one finite
  simplicial complex.

The corresponding operator-authorized verification capabilities are
`matrix.normal_form.smith.certified.verify` and
`topology.simplicial_homology.integral.verify`. Producers remain capped at
`COMPUTED`; only an accepted independent replay can return `VERIFIED`.

## Portfolio boundary

`matrix.normal_form.smith.compute` remains the lightweight, diagonal-only
matrix outcome. It explicitly reports that transformations are unavailable.
The certified operation is separate because full basis changes are a larger
artifact, have tighter input limits, and create different proof obligations.
It does not infer a cokernel, homology group, or preferred downstream use.

The integral topology operation is also separate from
`topology.simplicial_homology.compute`. The latter exposes bases over one
bounded prime field. The integral operation exposes finitely generated abelian
groups, including torsion, and therefore requires integer basis changes and
different witnesses. Neither operation ranks or prescribes the other.

## Transformation-certified Smith normal form

For a nonempty integer matrix \(A\), the result contains matrices \(D,U,V\)
with the exact relation

\[
D=UAV.
\]

Here \(U\) and \(V\) are square unimodular matrices, so their determinants are
\(\pm1\). The nonzero diagonal entries of \(D\) are positive and form the
canonical divisibility chain

\[
d_1\mid d_2\mid\cdots\mid d_r.
\]

The certificate records the source matrix, all three result matrices, the
rank, invariant factors, determinant claims, relation, transformation scope,
and normalization convention. This keeps the complete row and column basis
changes visible instead of retaining only the diagonal.

The request is bounded to 1–16 rows, 1–16 columns, canonical decimal integer
strings, and 32 input digits per entry. Certificate matrices support explicit
zero dimensions for reuse in chain-complex witnesses. Every output integer is
bounded to 32,768 digits. Complete request validation occurs before computation
or artifact writes.

The producer delegates the Smith decomposition to SymPy's
`smith_normal_decomp` over `ZZ` and has no optional-provider availability gate.
The canonical diagonal \(D\) and the invariant factors \(d_1,\ldots,d_r\) are
mathematical invariants: they are determined by the determinantal divisors of
\(A\) and do not depend on the backend. The unimodular transformations
\(U\) and \(V\), and every representative derived from them (kernel bases,
cycle coordinates, generator coordinates, bounding chains), are deterministic
for the pinned SymPy version but are **not** canonical: a different Smith
backend or SymPy release may produce different \(U,V\) that satisfy the same
relation \(D=UAV\). Compatibility is therefore semantic—the relation, both
unimodular determinants, and the positive divisibility diagonal—rather than
byte-identical transformations. The producer fail-closed checks verify all
three before returning, and the independent checker replays them without
calling the producer.

Python-FLINT 0.9.0's Python API exposes a diagonal-only `snf()` operation
even though current FLINT C documentation also describes
`fmpz_mat_snf_transform`. FLINT and SymPy are independent test oracles for
invariant factors, not hidden runtime dependencies of the checker.

## Integral simplicial homology

The input is the canonical materialized finite complex described by
[Finite simplicial topology](../capabilities/finite-math/finite-simplicial-topology.md), plus an explicit
`REDUCED` or `UNREDUCED` convention. Every simplex uses lexicographic vertex
orientation. For each dimension \(k\), the result exposes:

- the dimensions and ranks of \(C_k\), \(\partial_k\), and
  \(\operatorname{im}\partial_{k+1}\);
- the free rank and torsion invariant factors of \(H_k\);
- free and torsion cycle generators in the canonical \(k\)-simplex basis;
- cycle coordinates in an explicit kernel basis;
- a bounding \((k+1)\)-chain for every torsion generator; and
- Smith certificates for the outgoing boundary and the incoming boundary
  expressed in kernel coordinates.

If

\[
U_k\partial_kV_k=D_k
\]

has rank \(r_k\), the last columns of \(V_k\) form the declared integral
kernel basis \(K_k\). The producer records the exact factorization

\[
\partial_{k+1}=K_kQ_k
\]

and a second certificate

\[
\widehat U_kQ_k\widehat V_k=\widehat D_k.
\]

The nonunit entries of \(\widehat D_k\) are the torsion orders. Each reported
free or torsion coordinate \(y\) is bound by
\(\widehat U_ky=e_i\), and its simplex-basis cycle is \(K_ky\). For torsion
order \(d_i\), the recorded bounding chain \(b_i\) additionally satisfies

\[
\partial_{k+1}b_i=d_iK_ky.
\]

Thus the result is not merely a tuple of Betti numbers and abstract invariant
factors: the cycles, coordinate changes, quotient decomposition, and torsion
bounds are first-class inspectable artifacts.

Reduced degree-zero homology uses the augmentation kernel. It does not invent
or store a negative-dimensional simplex.

### Integral-specific bounds

The general topology materialization limits still apply. Integral homology has
tighter certificate limits:

| Quantity | Limit |
| --- | ---: |
| Simplices in one chain group | 16 |
| Sum of all chain-group ranks | 32 |
| Cells in one dense boundary shape | 256 |
| Decimal digits in an integral homology result integer | 256 |

These limits are independently enforced by the public request/result
contracts and the checker. Artifact-budget regressions cover a maximum-shape
certified Smith request and the 31-simplex triangulation of
\(\mathbb{RP}^2\).

## Independent verification

The certified-Smith checker strictly parses canonical integer encodings,
recomputes both matrix products, computes both determinants with a
fraction-free Bareiss algorithm, and checks the positive divisibility
diagonal. It never calls the producer's reduction implementation.

The integral-homology checker first reconstructs every oriented integer
boundary from the passive canonical complex. It then checks, in every
dimension:

1. the outgoing source and complete Smith relation;
2. the kernel basis from the certified right transformation;
3. the exact factorization of the incoming boundary through that kernel;
4. the second Smith relation and all free and torsion ranks;
5. every generator's kernel and simplex-basis coordinates; and
6. every torsion order and bounding-chain equation.

Replay binds the exact input and result artifacts, semantics, candidate digest,
witness format, and checker identity. Malformed data, unsupported scope,
timeout, cancellation, arithmetic failure, or a false relation returns
`UNKNOWN` and creates no verification record. A failed replay is not evidence
for an opposite mathematical claim.

## Nonclaims

This family does not provide persistent homology, a homology ring, cup
products, canonical generators independent of the declared simplex
orientation, manifold recognition, a preferred proof strategy, or conclusions
beyond the supplied bounded complex. SymPy is the producer backend for the
Smith decomposition; it is not an authoritative checker. The independent
checker uses only standard-library integer arithmetic and never imports the
producer or SymPy.

## Primary references

- [FLINT integer-matrix normal-form documentation](https://flintlib.org/doc/fmpz_mat.html#normal-forms)
- [Python-FLINT integer-matrix API](https://python-flint.readthedocs.io/en/latest/fmpz_mat.html)
- [SymPy normal-form API](https://docs.sympy.org/latest/modules/matrices/normalforms.html)
- [Hatcher, *Algebraic Topology*, Chapter 2](https://pi.math.cornell.edu/~hatcher/AT/ATch2.pdf)
