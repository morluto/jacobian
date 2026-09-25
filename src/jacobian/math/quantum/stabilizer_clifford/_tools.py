"""Catalog operation for exact Clifford transport of stabilizer groups."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.quantum._models import ExactStabilizerGroup
from jacobian.math.quantum.stabilizer_clifford._models import (
    StabilizerCliffordTransportRequest,
)
from jacobian.math.quantum.stabilizer_clifford.operations import (
    conjugate_stabilizer_group,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="quantum.stabilizer.clifford_gate.conjugate.compute",
        title="Conjugate an exact stabilizer group by one Clifford gate",
        description=(
            "Apply one H, S, or directed CNOT by exact conjugation to a typed "
            "phase-consistent stabilizer group. Return its exact independent "
            "generators with the same ordered register and phases."
        ),
        request_type=StabilizerCliffordTransportRequest,
        result_type=ExactStabilizerGroup,
        run=conjugate_stabilizer_group,
        tags=("quantum", "stabilizer", "clifford", "conjugation", "exact"),
        examples=(
            OperationExample(
                name="hadamard_transports_zero_stabilizer_to_plus",
                description="H Z H† = X on the named register qubit.",
                input={
                    "group": {
                        "register": {"qubit_ids": ["q0"]},
                        "generators": [
                            {
                                "phase_free": {
                                    "register": {"qubit_ids": ["q0"]},
                                    "x_bits": [0],
                                    "z_bits": [1],
                                },
                                "phase": 0,
                            }
                        ],
                    },
                    "gate": "H",
                    "qubits": ["q0"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
