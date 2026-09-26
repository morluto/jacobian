# Exact elementary Clifford conjugation of Pauli operators

[Operation references](index.md) · [Tool surface](../tools.md)

`quantum.pauli.clifford_gate.conjugate.compute` accepts one exact qubit Pauli
`i^r X^x Z^z` and one H, S, or directed CNOT gate on its explicitly ordered
register. It returns the exact conjugate `U P U†` as another `ExactQubitPauli`
with the same register. Its convention is `Y = iXZ`, with the global phase in
`Z/4Z`.

The local phase rules are:

- H swaps the selected `x,z` bits and adds phase `2xz`.
- S maps `(x,z)` to `(x,z+x)` and adds phase `x`.
- CNOT from `c` to `t` maps `x_t` to `x_t+x_c`, `z_c` to `z_c+z_t`, and
  leaves the phase unchanged.

Arithmetic is exact modulo two for coordinates and modulo four for phase. The
operation admits the full register scan and serialized result before allocating
coordinate copies; its request envelope is at most 32 qubits, one gate, 4,096
work units, and 16,384 compact JSON result bytes. This is one gate action, not
a gate-sequence or circuit interface. It does not claim to represent or compose
general Clifford automorphisms.
