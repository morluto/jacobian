"""Typed bounded sequences of elementary Clifford gates on qubit registers."""

from jacobian.math.quantum.stabilizer_clifford_sequence._models import (
    CliffordGate,
    StabilizerCliffordSequence,
)
from jacobian.math.quantum.stabilizer_clifford_sequence.operations import (
    apply_stabilizer_clifford_sequence,
    compose_stabilizer_clifford_sequences,
)

__all__ = [
    "CliffordGate",
    "StabilizerCliffordSequence",
    "apply_stabilizer_clifford_sequence",
    "compose_stabilizer_clifford_sequences",
]
