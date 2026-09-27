# Quantum stabilizer error equivalence

`quantum.stabilizer.error_equivalence.compute` compares two phase-free qubit
Pauli errors on the same ordered register modulo a supplied isotropic check
space. It returns their binary difference and whether that difference belongs
to the check-space row span. A true result means the errors have the same
logical action up to scalar phase.

Equal syndromes alone do not imply this relation: syndrome equality places the
difference in the symplectic orthogonal `S^perp`, while stabilizer equivalence
requires it to lie in `S`. A nonzero class in `S^perp/S` is a nontrivial logical
Pauli. The operation checks isotropy, canonicalizes the input row span over
GF(2), and decides membership by exact row reduction.

Requests admit at most 32 labelled qubits and 64 check rows. The operation
returns no phase claim, since scalar phases do not change the phase-free logical
error coset. The definitions follow Gottesman's stabilizer formalism, Chapter
3: [*Stabilizer Codes and Quantum Error Correction*](https://arxiv.org/abs/quant-ph/9705052).
