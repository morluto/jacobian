# Stabilizer syndrome maps

`quantum.stabilizer.syndrome_map.compute` returns the full binary linear map

\[
  \mathbb F_2^{2n} \longrightarrow S^*, \qquad
  e \longmapsto (\langle g_i,e\rangle)_i,
\]

where `S` is the supplied isotropic check space and `g_i` ranges over its
canonical independent RREF basis. The domain coordinates are flattened in the
order `[x_0,...,x_(n-1),z_0,...,z_(n-1)]`; all coordinates are phase-free and
over the fixed field `GF(2)`. Since
`<g,e> = x_g·z_e + z_g·x_e`, a check row `(x_g|z_g)` contributes the map row
`(z_g|x_g)`. The returned `NormalizerResult` is the exact kernel `S^perp`.

The result retains the supplied check space and the canonical check axis used
for the codomain dual coordinates. It records the field order, domain and
codomain dimensions, rank, kernel and fiber dimensions, and exact fiber
cardinality as a canonical decimal string for `2^fiber_dimension`. For an
isotropic check space of rank `r`, the map is surjective onto `S*`, so its rank
is `r`, its kernel dimension is `2n-r`, and each fiber has `2^(2n-r)` elements.
No dense Pauli operator or
stabilizer-group enumeration is constructed. The input has no Pauli phase or
code-eigenvalue character, so the result describes commutation syndromes and
does not choose a stabilizer state or joint eigenspace.

Admission supports at most 32 qubits and 64 input check rows. It bounds
pairwise isotropy checks, both binary row reductions, kernel and matrix output,
the exact fiber exponent, and a conservative 2,000,000-byte compact JSON result
before canonical/kernel expansion.

The convention follows the stabilizer symplectic formalism in Gottesman's
dissertation, [*Stabilizer Codes and Quantum Error Correction*, Chapter 3](https://arxiv.org/abs/quant-ph/9705052).
