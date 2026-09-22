"""Public declaration for exact stabilizer check-space canonicalization."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quantum._models import (
    CheckSpaceCanonicalizeRequest,
    CheckSpaceCanonicalizeResult,
    CheckSpaceValue,
    NormalizerResult,
    PauliInverseRequest,
    PauliInverseResult,
    PauliPairingRequest,
    PauliPairingResult,
    PauliProductRequest,
    PauliProductResult,
)
from jacobian.math.quantum.operations import (
    canonicalize_check_space,
    pauli_inverse,
    pauli_multiply,
    pauli_pairing,
    stabilizer_normalizer,
)


def _run_canonicalize_check_space(
    request: CheckSpaceCanonicalizeRequest,
) -> CheckSpaceCanonicalizeResult:
    return canonicalize_check_space(request.qubit_ids, request.generators)


_REGISTER = {"qubit_ids": ["q0"]}
_X = {"register": _REGISTER, "x_bits": [1], "z_bits": [0]}
_Z = {"register": _REGISTER, "x_bits": [0], "z_bits": [1]}


def _run_product(request: PauliProductRequest) -> PauliProductResult:
    return pauli_multiply(request.left, request.right)


def _run_pairing(request: PauliPairingRequest) -> PauliPairingResult:
    return pauli_pairing(request.left, request.right)


def _run_inverse(request: PauliInverseRequest) -> PauliInverseResult:
    return pauli_inverse(request.pauli)


TOOLS = (
    MathTool(
        operation_id="quantum.pauli.inverse.compute",
        title="Invert an exact register-bound qubit Pauli",
        description="Compute the exact inverse phase and vector of one Pauli under the fixed i^r X^x Z^z convention; the returned value remains bound to its ordered qubit register.",
        request_type=PauliInverseRequest,
        result_type=PauliInverseResult,
        run=_run_inverse,
        tags=("quantum", "pauli", "inverse", "exact"),
        examples=(
            OperationExample(
                name="inverse_of_x",
                description="Compute the exact inverse of X; the request's left Pauli must be register-bound.",
                input={"pauli": {"phase_free": _X, "phase": 0}},
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.pauli.multiply.compute",
        title="Multiply exact register-bound qubit Paulis",
        description=(
            "Multiply exact qubit Paulis under the fixed convention "
            "i^r X^x Z^z, returning the exact cocycle phase and register-bound "
            "phase-free product. Both operands must use the identical ordered register."
        ),
        request_type=PauliProductRequest,
        result_type=PauliProductResult,
        run=_run_product,
        tags=("quantum", "pauli", "phase", "exact"),
        discovery_terms=(
            "Pauli multiplication",
            "Pauli cocycle",
            "exact qubit operator product",
        ),
        examples=(
            OperationExample(
                name="x_times_z",
                description="Multiply X and Z on one qubit; both exact Paulis must use the same ordered register and phase convention.",
                input={
                    "left": {"phase_free": _X, "phase": 0},
                    "right": {"phase_free": _Z, "phase": 0},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.pauli.symplectic_pairing.compute",
        title="Compute the exact Pauli symplectic pairing",
        description="Compute the GF(2) symplectic pairing and commutation relation of two phase-free Paulis on one ordered register; equal pairing zero is exact commutation.",
        request_type=PauliPairingRequest,
        result_type=PauliPairingResult,
        run=_run_pairing,
        tags=("quantum", "pauli", "symplectic", "exact"),
        examples=(
            OperationExample(
                name="x_z_anticommutation",
                description="Compute the nonzero X/Z symplectic pairing; both phase-free values must share one register.",
                input={"left": _X, "right": _Z},
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.normalizer.compute",
        title="Compute the exact stabilizer symplectic orthogonal",
        description=(
            "Compute S-perp for a register-bound isotropic binary stabilizer check "
            "space, retaining canonical bases and the exact quotient dimension "
            "dim(S-perp/S). Non-isotropic authored check spaces are rejected."
        ),
        request_type=CheckSpaceValue,
        result_type=NormalizerResult,
        run=stabilizer_normalizer,
        tags=("quantum", "stabilizer", "normalizer", "symplectic", "exact"),
        examples=(
            OperationExample(
                name="single_qubit_z_normalizer",
                description="Compute the normalizer of the one-qubit Z check; the check space must be isotropic on its retained register.",
                input={
                    "register": _REGISTER,
                    "basis": [{"register": _REGISTER, "x_bits": [0], "z_bits": [1]}],
                },
            ),
        ),
    ),
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
