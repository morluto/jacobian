# Exact qubit stabilizer groups

`quantum.stabilizer.exact_group.from_generators.compute` accepts exact Pauli
values under the convention

\[
P = i^r X^x Z^z,
\]

on one ordered qubit register. It returns an independent exact generating
family for the subgroup they generate, provided every generator is Hermitian,
all pairs commute, and every binary dependency multiplies to (+I). Dependent
rows with product (+I) are removed. A dependency producing (-I), a
non-Hermitian generator, an anticommuting pair, or a register mismatch is
rejected as invalid input.

The generator family need not be independent. The operation scans all pairs,
then incrementally reduces each phase-free vector over GF(2), multiplying the
corresponding exact Paulis as it eliminates coordinates. Consequently a zero
vector is checked with its actual scalar phase, rather than only as a relation
among binary rows. The returned independent family is selected deterministically
in input order and retains the exact phases of those rows; it is not a canonical
presentation of the subgroup.

The admitted envelope is at most 32 qubits and 64 input generators. The
operation does not enumerate the generated group or construct a dense
Hilbert-space operator. It establishes an exact phase-consistent group
presentation, not a selected common eigenspace, state vector, or code projector.
