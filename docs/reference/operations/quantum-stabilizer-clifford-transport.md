# Exact Clifford transport of stabilizer groups

`quantum.stabilizer.clifford_gate.conjugate.compute` conjugates one
phase-consistent exact stabilizer group by one Hadamard (`H`), phase (`S`), or
directed controlled-not (`CNOT`) gate. It returns an exact independent
generating family for the group `U S U†`, with the same ordered register.

The request supplies an `ExactStabilizerGroup`, a gate name, and one target
qubit for `H`/`S` or an ordered `(control, target)` pair for `CNOT`. Each
returned generator is a phase-lifted Pauli under the convention
`i^r X^x Z^z`. In particular, `Y=iXZ` and signs are preserved by the phase
updates. This is a one-gate action; callers may compose unchanged results by
calling the operation again.

The operation revalidates the supplied group's exact generator relations,
canonicalizes its independent generating family, and applies the gate. The
Clifford action preserves Hermiticity, commutation, and independence, so the
result is again an exact stabilizer group. The empty generating family is the
trivial group and remains empty under every supported gate.

Admission bounds source relation checks, register and generator validation,
gate transformation work, and serialized result size before exact
canonicalization or transformed-row allocation. The current envelope supports
at most 32 qubits, 64 source generators, 1,000,000 work units, and 65,536
compact result bytes. These bounds are independent of the exponentially larger
group order: the operation never enumerates group elements.

The convention and preservation law follow the stabilizer and Clifford
formalism in Gottesman's primary account,
[*Stabilizer Codes and Quantum Error Correction*](https://arxiv.org/abs/quant-ph/9705052).
