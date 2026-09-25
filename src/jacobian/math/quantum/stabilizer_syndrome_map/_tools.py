"""Catalog manifest for exact stabilizer syndrome maps."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.quantum.stabilizer_syndrome_map._models import (
    SyndromeMapRequest,
    SyndromeMapResult,
)
from jacobian.math.quantum.stabilizer_syndrome_map.operations import syndrome_map


def _run(request: SyndromeMapRequest) -> SyndromeMapResult:
    return syndrome_map(request.check_space)


TOOLS: MathTools = (
    MathTool(
        operation_id="quantum.stabilizer.syndrome_map.compute",
        title="Compute the exact stabilizer syndrome map",
        description=(
            "Return the exact GF(2)-linear map from phase-free Pauli coordinates "
            "[x|z] to the dual coordinates of the canonical independent check "
            "basis, together with its S-perp kernel, rank, and fiber dimensions. "
            "The map is source- and register-bound, uses at most 32 qubits and 64 "
            "input rows, and admits output and work before kernel expansion."
        ),
        request_type=SyndromeMapRequest,
        result_type=SyndromeMapResult,
        run=_run,
        tags=("quantum", "stabilizer", "syndrome", "linear-map", "exact"),
        examples=(
            OperationExample(
                name="one_zz_check_map",
                description=(
                    "For check ZZ on (q0,q1), the row is [1,1,0,0] in [x0,x1,z0,z1] coordinates."
                ),
                input={
                    "check_space": {
                        "register": {"qubit_ids": ["q0", "q1"]},
                        "basis": [
                            {
                                "register": {"qubit_ids": ["q0", "q1"]},
                                "x_bits": [0, 0],
                                "z_bits": [1, 1],
                            }
                        ],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
