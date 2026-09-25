"""Public operation declarations for finite Clifford sequence composition."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.quantum._models import ExactStabilizerGroup
from jacobian.math.quantum.stabilizer_clifford_sequence._models import (
    CliffordSequenceCompositionRequest,
    StabilizerCliffordSequence,
    StabilizerCliffordSequenceApplyRequest,
)
from jacobian.math.quantum.stabilizer_clifford_sequence.operations import (
    apply_stabilizer_clifford_sequence,
    compose_stabilizer_clifford_sequences,
)

_REGISTER = {"qubit_ids": ["q0", "q1"]}
_Z0 = {
    "phase_free": {"register": _REGISTER, "x_bits": [0, 0], "z_bits": [1, 0]},
    "phase": 0,
}

TOOLS: MathTools = (
    MathTool(
        operation_id="quantum.stabilizer.clifford_sequence.compose.compute",
        title="Compose finite Clifford gate sequences",
        description=(
            "Compose two register-bound finite sequences of H, S, and directed "
            "CNOT gates. The left sequence acts first and the right sequence "
            "acts second. The returned typed sequence preserves that exact order."
        ),
        request_type=CliffordSequenceCompositionRequest,
        result_type=StabilizerCliffordSequence,
        run=compose_stabilizer_clifford_sequences,
        tags=("quantum", "stabilizer", "clifford", "composition", "exact"),
        examples=(
            OperationExample(
                name="compose_h_then_cnot",
                description="Apply H to q0, then CNOT with q0 as control and q1 as target.",
                input={
                    "left": {
                        "register": _REGISTER,
                        "gates": [{"gate": "H", "qubits": ["q0"]}],
                    },
                    "right": {
                        "register": _REGISTER,
                        "gates": [{"gate": "CNOT", "qubits": ["q0", "q1"]}],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.clifford_sequence.apply.compute",
        title="Apply a finite Clifford gate sequence to a stabilizer group",
        description=(
            "Conjugate an exact stabilizer group by a bounded typed sequence of "
            "H, S, and directed CNOT gates. Return the exact transported group "
            "with its ordered register and Pauli phases. The empty sequence is "
            "the identity."
        ),
        request_type=StabilizerCliffordSequenceApplyRequest,
        result_type=ExactStabilizerGroup,
        run=apply_stabilizer_clifford_sequence,
        tags=("quantum", "stabilizer", "clifford", "transport", "exact"),
        examples=(
            OperationExample(
                name="hadamard_then_phase_on_checks",
                description="Transport the Z0 check first by H(q0), then by S(q0).",
                input={
                    "group": {"register": _REGISTER, "generators": [_Z0]},
                    "sequence": {
                        "register": _REGISTER,
                        "gates": [
                            {"gate": "H", "qubits": ["q0"]},
                            {"gate": "S", "qubits": ["q0"]},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
