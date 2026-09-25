"""Public declaration for phase-free Pauli vector addition."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quantum._models import PhaseFreeQubitPauli
from jacobian.math.quantum.phase_free_add._models import (
    PhaseFreePauliAdditionRequest,
)
from jacobian.math.quantum.phase_free_add.operations import add_phase_free_paulis

_REGISTER = {"qubit_ids": ["q0"]}


def _run_add(request: PhaseFreePauliAdditionRequest) -> PhaseFreeQubitPauli:
    return add_phase_free_paulis(request.left, request.right)


TOOLS = (
    MathTool(
        operation_id="quantum.pauli.phase_free.add.compute",
        title="Add phase-free qubit Paulis",
        description=(
            "Add the binary X and Z coordinates of two phase-free qubit Pauli "
            "values on the identical ordered register. The result is their "
            "product modulo scalar phase."
        ),
        request_type=PhaseFreePauliAdditionRequest,
        result_type=PhaseFreeQubitPauli,
        run=_run_add,
        tags=("quantum", "pauli", "phase-free", "addition", "exact"),
        discovery_terms=(
            "Pauli product modulo phase",
            "binary symplectic vector addition",
            "add phase-free Paulis",
        ),
        examples=(
            OperationExample(
                name="cancel_equal_paulis",
                description="Multiply X by X modulo phase on one qubit.",
                input={
                    "left": {
                        "register": _REGISTER,
                        "x_bits": [1],
                        "z_bits": [0],
                    },
                    "right": {
                        "register": _REGISTER,
                        "x_bits": [1],
                        "z_bits": [0],
                    },
                },
            ),
        ),
    ),
)
