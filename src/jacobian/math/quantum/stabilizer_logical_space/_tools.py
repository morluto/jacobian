"""Public operation declaration for the logical Pauli quotient space."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.quantum._models import CheckSpaceValue
from jacobian.math.quantum.stabilizer_logical_space._models import LogicalPauliSpace
from jacobian.math.quantum.stabilizer_logical_space.operations import (
    logical_pauli_space,
)

_REGISTER = {"qubit_ids": ["q0", "q1"]}

TOOLS: MathTools = (
    MathTool(
        operation_id="quantum.stabilizer.logical_pauli_space.compute",
        title="Construct the phase-free logical Pauli quotient space",
        description=(
            "Construct the exact GF(2) quotient S-perp/S for an isotropic "
            "register-bound stabilizer check space. Return source-bound "
            "inclusion, projection, ambient embedding and representative-lift "
            "maps, together with the induced nondegenerate symplectic form."
        ),
        request_type=CheckSpaceValue,
        result_type=LogicalPauliSpace,
        run=logical_pauli_space,
        tags=("quantum", "stabilizer", "logical", "quotient", "symplectic", "exact"),
        discovery_terms=(
            "logical Pauli quotient",
            "stabilizer logical space",
            "symplectic quotient S-perp/S",
        ),
        examples=(
            OperationExample(
                name="one_logical_qubit_after_one_z_check",
                description="Compute S-perp/S for a Z check on q0 of a two-qubit register.",
                input={
                    "register": _REGISTER,
                    "basis": [
                        {
                            "register": _REGISTER,
                            "x_bits": [0, 0],
                            "z_bits": [1, 0],
                        }
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
