"""Public declaration for exact qubit Pauli register relabelling."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quantum._models import ExactQubitPauli
from jacobian.math.quantum.pauli_relabel._models import PauliRelabelRequest
from jacobian.math.quantum.pauli_relabel.operations import relabel_pauli

_SOURCE = {"qubit_ids": ["left", "right"]}
_TARGET = {"qubit_ids": ["second", "first"]}


def _run_relabel(request: PauliRelabelRequest) -> ExactQubitPauli:
    return relabel_pauli(request.pauli, request.relabeling)


TOOLS = (
    MathTool(
        operation_id="quantum.pauli.relabel_qubits.compute",
        title="Relabel an exact Pauli across qubit registers",
        description=(
            "Transport an exact qubit Pauli along an explicit bijection from its "
            "ordered source register to an ordered target register. "
            "target_ids_in_source_order gives the image of each source axis; "
            "the exact phase and Pauli action are preserved."
        ),
        request_type=PauliRelabelRequest,
        result_type=ExactQubitPauli,
        run=_run_relabel,
        tags=("quantum", "pauli", "register", "relabel", "exact"),
        discovery_terms=(
            "qubit register permutation",
            "Pauli coordinate transport",
            "relabel tensor factors",
        ),
        examples=(
            OperationExample(
                name="reorder_pauli_axes",
                description="Move an exact Pauli to a reordered target register by an explicit axis bijection.",
                input={
                    "pauli": {
                        "phase_free": {
                            "register": _SOURCE,
                            "x_bits": [1, 0],
                            "z_bits": [0, 1],
                        },
                        "phase": 3,
                    },
                    "relabeling": {
                        "source_register": _SOURCE,
                        "target_register": _TARGET,
                        "target_ids_in_source_order": ["first", "second"],
                    },
                },
            ),
        ),
    ),
)
