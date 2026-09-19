"""Public declaration for exact stabilizer check-space canonicalization."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quantum._models import (
    CheckSpaceCanonicalizeRequest,
    CheckSpaceCanonicalizeResult,
)
from jacobian.math.quantum.operations import canonicalize_check_space


def _run_canonicalize_check_space(
    request: CheckSpaceCanonicalizeRequest,
) -> CheckSpaceCanonicalizeResult:
    return canonicalize_check_space(request.qubit_ids, request.generators)


TOOLS = (
    MathTool(
        operation_id="stabilizer.check_space.canonicalize",
        title="Canonicalize a binary stabilizer check matrix",
        description=(
            "For a binary check matrix over a labelled qubit register, return "
            "the canonical GF(2) RREF basis with rank and the isotropic "
            "decision, or an explicit non-commuting witness pair. Phase-free "
            "Paulis commute iff their symplectic pairing x.z' + z.x' is zero "
            "mod 2; zero rows contribute no pivot. The canonical basis depends "
            "only on the GF(2) row space, while input row IDs use one canonical "
            "strictly ordered presentation."
        ),
        request_type=CheckSpaceCanonicalizeRequest,
        result_type=CheckSpaceCanonicalizeResult,
        run=_run_canonicalize_check_space,
        tags=("stabilizer", "pauli", "symplectic", "exact"),
        discovery_terms=(
            "stabilizer check matrix RREF",
            "Pauli commutation symplectic pairing",
            "isotropic stabilizer subspace",
        ),
        examples=(
            OperationExample(
                name="bell_pair_checks",
                description=(
                    "Canonicalize the two commuting Bell-pair checks XX and ZZ "
                    "on two qubits; generator rows must match the register "
                    "length."
                ),
                input={
                    "qubit_ids": ["q0", "q1"],
                    "generators": [
                        {
                            "row_id": "xx",
                            "x_bits": [1, 1],
                            "z_bits": [0, 0],
                        },
                        {
                            "row_id": "zz",
                            "x_bits": [0, 0],
                            "z_bits": [1, 1],
                        },
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
