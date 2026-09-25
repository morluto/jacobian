"""Catalog declaration for elementary exact Clifford Pauli conjugation."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.quantum._models import ExactQubitPauli
from jacobian.math.quantum.pauli_clifford._models import (
    PauliCliffordConjugationRequest,
)
from jacobian.math.quantum.pauli_clifford.operations import conjugate_pauli

TOOLS: MathTools = (
    MathTool(
        operation_id="quantum.pauli.clifford_gate.conjugate.compute",
        title="Conjugate an exact Pauli by one Clifford gate",
        description=(
            "Apply one H, S, or directed CNOT gate by exact conjugation to an "
            "i^r X^x Z^z Pauli on its bound ordered qubit register. Return the "
            "exact phase-lifted Pauli. Gate roles use register labels; this "
            "operation applies one gate and accepts no circuit sequence."
        ),
        request_type=PauliCliffordConjugationRequest,
        result_type=ExactQubitPauli,
        run=conjugate_pauli,
        tags=("quantum", "pauli", "clifford", "conjugation", "exact"),
        examples=(
            OperationExample(
                name="hadamard_maps_x_to_z",
                description="H X H† = Z on the named register qubit.",
                input={
                    "pauli": {
                        "phase_free": {
                            "register": {"qubit_ids": ["q0"]},
                            "x_bits": [1],
                            "z_bits": [0],
                        },
                        "phase": 0,
                    },
                    "gate": "H",
                    "qubits": ["q0"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
