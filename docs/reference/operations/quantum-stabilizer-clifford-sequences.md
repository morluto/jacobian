# Finite Clifford sequences on stabilizer groups

`quantum.stabilizer.clifford_sequence.compose.compute` and
`quantum.stabilizer.clifford_sequence.apply.compute` use the immutable
`StabilizerCliffordSequence` value. It binds an ordered qubit register to up to
128 typed gates, each of which is `H`, `S`, or a directed `CNOT` on labels in
that register. The empty sequence is the identity. The sequence is finite
mathematical input, not a free-form circuit string or a persistent execution
session.

Composition accepts two sequences on the same ordered register. It appends the
right sequence after the left sequence: for `left=(g1, g2)` and `right=(g3)`,
the combined action applies `g1`, then `g2`, then `g3`. Empty sequences are left
and right identities. The result remains an ordinary typed sequence that can
be serialized and consumed unchanged.

Application accepts an exact stabilizer group on the sequence register and
returns its exact conjugate. Generator phases use Jacobian's fixed
`i^r X^x Z^z` convention and are updated at each gate. The group is validated
and canonicalized once, then each generator is transported without building a
dense `2^n × 2^n` operator or enumerating group elements. The empty gate
sequence preserves the group as a subgroup, including the trivial group.

Each sequence is capped at 128 gates. Application checks the source group
relations, gate-axis resolution, per-generator transformations, and result
bytes before canonicalization or transformed-row allocation. Its envelope is
1,500,000 work units and 65,536 compact result bytes, in addition to the
register and generator limits of the exact-group operation. Composition
preflights copying work, result size (512,000 bytes), and the combined gate
count before concatenating the sequences. Exceeding any bound is a resource
admission error; it does not indicate a mathematical result.

The convention and composition law follow the stabilizer/Clifford formalism
in Gottesman, [*Stabilizer Codes and Quantum Error Correction*](https://arxiv.org/abs/quant-ph/9705052).
