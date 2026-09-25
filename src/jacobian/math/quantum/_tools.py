"""Public declaration for exact stabilizer check-space canonicalization."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quantum._models import (
    CheckSpaceCanonicalizeRequest,
    CheckSpaceCanonicalizeResult,
    CheckSpaceValue,
    CSSCheckSpaceRequest,
    CSSCheckSpaceResult,
    CSSCheckSpaceValue,
    CSSDistanceResult,
    CSSLogicalPauliFrame,
    ExactStabilizerGroup,
    ExactStabilizerGroupRequest,
    LogicalPauliFrame,
    NormalizerResult,
    PauliFamilyCommutationRequest,
    PauliFamilyCommutationResult,
    PauliFromLabelsRequest,
    PauliFromLabelsResult,
    PauliInverseRequest,
    PauliInverseResult,
    PauliPairingRequest,
    PauliPairingResult,
    PauliProductRequest,
    PauliProductResult,
    PauliToLabelsRequest,
    PauliToLabelsResult,
    StabilizerCodeRequest,
    StabilizerCodeValue,
    StabilizerDistanceResult,
    StabilizerErrorEquivalenceRequest,
    StabilizerErrorEquivalenceResult,
    StabilizerSyndromeRequest,
    StabilizerSyndromeResult,
)
from jacobian.math.quantum.operations import (
    canonicalize_check_space,
    css_check_space,
    css_exact_distance,
    css_logical_pauli_frame,
    pauli_family_commutation_matrix,
    pauli_from_labels,
    pauli_inverse,
    pauli_multiply,
    pauli_pairing,
    pauli_to_labels,
    stabilizer_code_compute,
    stabilizer_error_equivalence,
    stabilizer_exact_distance,
    stabilizer_group_from_generators,
    stabilizer_logical_frame,
    stabilizer_normalizer,
    stabilizer_syndrome,
)


def _run_canonicalize_check_space(
    request: CheckSpaceCanonicalizeRequest,
) -> CheckSpaceCanonicalizeResult:
    return canonicalize_check_space(request.qubit_ids, request.generators)


def _run_css_check_space(request: CSSCheckSpaceRequest) -> CSSCheckSpaceResult:
    return css_check_space(request)


def _run_css_logical_frame(value: CSSCheckSpaceValue) -> CSSLogicalPauliFrame:
    return css_logical_pauli_frame(value)


def _run_css_distance(value: CSSCheckSpaceValue) -> CSSDistanceResult:
    return css_exact_distance(value)


def _run_stabilizer_distance(value: CheckSpaceValue) -> StabilizerDistanceResult:
    return stabilizer_exact_distance(value)


def _run_logical_frame(value: CheckSpaceValue) -> LogicalPauliFrame:
    return stabilizer_logical_frame(value)


_REGISTER = {"qubit_ids": ["q0"]}
_X = {"register": _REGISTER, "x_bits": [1], "z_bits": [0]}
_Z = {"register": _REGISTER, "x_bits": [0], "z_bits": [1]}


def _run_product(request: PauliProductRequest) -> PauliProductResult:
    return pauli_multiply(request.left, request.right)


def _run_pairing(request: PauliPairingRequest) -> PauliPairingResult:
    return pauli_pairing(request.left, request.right)


def _run_family_commutation(
    request: PauliFamilyCommutationRequest,
) -> PauliFamilyCommutationResult:
    return pauli_family_commutation_matrix(request)


def _run_inverse(request: PauliInverseRequest) -> PauliInverseResult:
    return pauli_inverse(request.pauli)


def _run_from_labels(request: PauliFromLabelsRequest) -> PauliFromLabelsResult:
    return pauli_from_labels(request)


def _run_to_labels(request: PauliToLabelsRequest) -> PauliToLabelsResult:
    return pauli_to_labels(request)


def _run_syndrome(
    request: StabilizerSyndromeRequest,
) -> StabilizerSyndromeResult:
    return stabilizer_syndrome(request.check_space, request.error)


def _run_error_equivalence(
    request: StabilizerErrorEquivalenceRequest,
) -> StabilizerErrorEquivalenceResult:
    return stabilizer_error_equivalence(
        request.check_space, request.left, request.right
    )


def _run_exact_stabilizer_group(
    request: ExactStabilizerGroupRequest,
) -> ExactStabilizerGroup:
    return stabilizer_group_from_generators(request)


TOOLS = (
    MathTool(
        operation_id="quantum.stabilizer.code.compute",
        title="Select a canonical stabilizer code eigenspace",
        description=(
            "Bind an exact commuting Hermitian Pauli group to a character given "
            "by one +1 or -1 eigenvalue per independent generator. Return the "
            "selected joint eigenspace as a canonical RREF group whose generators "
            "all have eigenvalue +1 on that code. With n physical qubits and "
            "rank r, it encodes n-r logical qubits and has Hilbert-space "
            "dimension 2^(n-r); no dense state or projector is formed."
        ),
        request_type=StabilizerCodeRequest,
        result_type=StabilizerCodeValue,
        run=stabilizer_code_compute,
        tags=("quantum", "stabilizer", "code-space", "character", "exact"),
        discovery_terms=(
            "stabilizer code eigenspace",
            "stabilizer eigenvalue character",
            "joint Pauli eigenspace",
        ),
        examples=(
            OperationExample(
                name="bell_state_code_space",
                description=(
                    "Select the Bell state stabilized by XX and ZZ; both exact "
                    "group generators must be independent commuting Hermitian Paulis."
                ),
                input={
                    "group": {
                        "register": {"qubit_ids": ["q0", "q1"]},
                        "generators": [
                            {
                                "phase_free": {
                                    "register": {"qubit_ids": ["q0", "q1"]},
                                    "x_bits": [1, 1],
                                    "z_bits": [0, 0],
                                },
                                "phase": 0,
                            },
                            {
                                "phase_free": {
                                    "register": {"qubit_ids": ["q0", "q1"]},
                                    "x_bits": [0, 0],
                                    "z_bits": [1, 1],
                                },
                                "phase": 0,
                            },
                        ],
                    },
                    "generator_eigenvalues": [1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.exact_group.from_generators.compute",
        title="Construct an exact qubit stabilizer group",
        description=(
            "Validate a supplied family of exact phase-lifted Pauli generators, "
            "then return an independent generating family for the same group. "
            "Every generator must be Hermitian, the generators must commute, "
            "and no binary dependency may multiply to a nonidentity scalar. "
            "Dependent rows multiplying to +I are accepted and removed."
        ),
        request_type=ExactStabilizerGroupRequest,
        result_type=ExactStabilizerGroup,
        run=_run_exact_stabilizer_group,
        tags=("quantum", "stabilizer", "pauli", "phase", "exact"),
        discovery_terms=(
            "phase-consistent stabilizer group generators",
            "reject stabilizer generators whose product is minus identity",
            "exact Pauli stabilizer subgroup",
        ),
        examples=(
            OperationExample(
                name="bell_stabilizer_group",
                description=(
                    "Construct the two-generator Bell stabilizer group; both "
                    "exact Hermitian checks commute and are independent."
                ),
                input={
                    "register": {"qubit_ids": ["q0", "q1"]},
                    "generators": [
                        {
                            "phase_free": {
                                "register": {"qubit_ids": ["q0", "q1"]},
                                "x_bits": [1, 1],
                                "z_bits": [0, 0],
                            },
                            "phase": 0,
                        },
                        {
                            "phase_free": {
                                "register": {"qubit_ids": ["q0", "q1"]},
                                "x_bits": [0, 0],
                                "z_bits": [1, 1],
                            },
                            "phase": 0,
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.logical_frame.compute",
        title="Compute a general stabilizer logical Pauli frame",
        description=(
            "Choose deterministic phase-free representatives of a symplectic "
            "basis of S-perp/S for any isotropic register-bound check space. "
            "The two returned k-element families pair as the logical X/Z "
            "basis; representatives may mix physical X and Z coordinates."
        ),
        request_type=CheckSpaceValue,
        result_type=LogicalPauliFrame,
        run=_run_logical_frame,
        tags=("quantum", "stabilizer", "logical", "quotient", "symplectic", "exact"),
        examples=(
            OperationExample(
                name="two_qubit_yi_check_logical_frame",
                description=(
                    "Compute a logical frame for the YI stabilizer on two qubits; "
                    "the supplied check space must be isotropic."
                ),
                input={
                    "register": {"qubit_ids": ["q0", "q1"]},
                    "basis": [
                        {
                            "register": {"qubit_ids": ["q0", "q1"]},
                            "x_bits": [1, 0],
                            "z_bits": [1, 0],
                        }
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.css_distance.compute",
        title="Compute exact CSS X and Z distances",
        description=(
            "Exhaustively find the minimum Hamming weight in ker(H_Z) outside "
            "row(H_X), and in ker(H_X) outside row(H_Z). Return one minimum "
            "register-order-tiebroken representative for each logical sector. "
            "The exact search is refused before enumeration when its complete "
            "candidate or work bound exceeds the operation envelope."
        ),
        request_type=CSSCheckSpaceValue,
        result_type=CSSDistanceResult,
        run=_run_css_distance,
        tags=("quantum", "stabilizer", "CSS", "distance", "exact", "exhaustive"),
        examples=(
            OperationExample(
                name="four_qubit_css_distance",
                description="Compute both CSS distances for the four-qubit code with XXXX and ZZZZ checks.",
                input={
                    "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                    "x_check_basis": [
                        {
                            "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                            "x_bits": [1, 1, 1, 1],
                            "z_bits": [0, 0, 0, 0],
                        }
                    ],
                    "z_check_basis": [
                        {
                            "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                            "x_bits": [0, 0, 0, 0],
                            "z_bits": [1, 1, 1, 1],
                        }
                    ],
                    "check_space": {
                        "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                        "basis": [
                            {
                                "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                                "x_bits": [1, 1, 1, 1],
                                "z_bits": [0, 0, 0, 0],
                            },
                            {
                                "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                                "x_bits": [0, 0, 0, 0],
                                "z_bits": [1, 1, 1, 1],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.distance.compute",
        title="Compute exact mixed-Pauli logical distance",
        description=(
            "Exhaustively compute the minimum weight of a Pauli in S-perp outside "
            "the stabilizer row space for a general isotropic check space, including "
            "mixed X/Z checks. Returns one register-bound minimum representative; "
            "k=0 returns no distance. Exact search is admitted only through 10 qubits."
        ),
        request_type=CheckSpaceValue,
        result_type=StabilizerDistanceResult,
        run=_run_stabilizer_distance,
        tags=("quantum", "stabilizer", "distance", "logical", "exact", "exhaustive"),
        discovery_terms=(
            "general stabilizer code distance",
            "mixed Pauli logical distance",
        ),
        examples=(
            OperationExample(
                name="two_qubit_y_check_distance",
                description="Compute exact distance for the isotropic YI stabilizer, with mixed Paulis allowed.",
                input={
                    "register": {"qubit_ids": ["q0", "q1"]},
                    "basis": [
                        {
                            "register": {"qubit_ids": ["q0", "q1"]},
                            "x_bits": [1, 0],
                            "z_bits": [1, 0],
                        }
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.css_logical_frame.compute",
        title="Compute a CSS logical Pauli frame",
        description=(
            "Compute deterministic X/Z representatives of the CSS quotient "
            "ker(H_Z)/row(H_X) and ker(H_X)/row(H_Z), dual under the binary "
            "symplectic pairing. The source CSS value and combined check space "
            "must retain matching register-bound roles."
        ),
        request_type=CSSCheckSpaceValue,
        result_type=CSSLogicalPauliFrame,
        run=_run_css_logical_frame,
        tags=("quantum", "stabilizer", "CSS", "logical", "quotient", "exact"),
        examples=(
            OperationExample(
                name="four_qubit_css_logical_frame",
                description="Find a logical X/Z symplectic frame for the four-qubit CSS code with XXXX and ZZZZ checks.",
                input={
                    "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                    "x_check_basis": [
                        {
                            "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                            "x_bits": [1, 1, 1, 1],
                            "z_bits": [0, 0, 0, 0],
                        }
                    ],
                    "z_check_basis": [
                        {
                            "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                            "x_bits": [0, 0, 0, 0],
                            "z_bits": [1, 1, 1, 1],
                        }
                    ],
                    "check_space": {
                        "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                        "basis": [
                            {
                                "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                                "x_bits": [1, 1, 1, 1],
                                "z_bits": [0, 0, 0, 0],
                            },
                            {
                                "register": {"qubit_ids": ["q0", "q1", "q2", "q3"]},
                                "x_bits": [0, 0, 0, 0],
                                "z_bits": [1, 1, 1, 1],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.css_check_space.compute",
        title="Construct a binary CSS stabilizer check space",
        description=(
            "Canonicalize binary X- and Z-check rows on one ordered qubit register. "
            "Return their exact isotropic combined check space, or an input-row "
            "witness to failure of H_X H_Z^T = 0."
        ),
        request_type=CSSCheckSpaceRequest,
        result_type=CSSCheckSpaceResult,
        run=_run_css_check_space,
        tags=("quantum", "stabilizer", "CSS", "symplectic", "exact"),
        examples=(
            OperationExample(
                name="binary_css_checks",
                description="Build a commuting CSS check space from orthogonal binary X/Z rows.",
                input={
                    "register": {"qubit_ids": ["q0", "q1", "q2"]},
                    "x_checks": [[1, 1, 0]],
                    "z_checks": [[1, 1, 0]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.stabilizer.error_equivalence.compute",
        title="Compare Pauli errors modulo stabilizers",
        description=(
            "Decide whether two phase-free Pauli errors differ by an element of "
            "an isotropic check space. Equal syndrome alone is not sufficient."
        ),
        request_type=StabilizerErrorEquivalenceRequest,
        result_type=StabilizerErrorEquivalenceResult,
        run=_run_error_equivalence,
        tags=("quantum", "stabilizer", "error-equivalence", "logical", "exact"),
        examples=(
            OperationExample(
                name="same_error_modulo_zz",
                description="Compare I and ZZ on a two-qubit register with ZZ as the check generator.",
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
                    },
                    "left": {
                        "register": {"qubit_ids": ["q0", "q1"]},
                        "x_bits": [0, 0],
                        "z_bits": [0, 0],
                    },
                    "right": {
                        "register": {"qubit_ids": ["q0", "q1"]},
                        "x_bits": [0, 0],
                        "z_bits": [1, 1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.pauli.qubit.from_labels.compute",
        title="Construct an exact qubit Pauli from labels",
        description=(
            "Convert one complete ordered I/X/Y/Z row and scalar phase to the "
            "register-bound i^r X^x Z^z convention. Each Y contributes one "
            "factor of i, and the result retains the full register axis."
        ),
        request_type=PauliFromLabelsRequest,
        result_type=PauliFromLabelsResult,
        run=_run_from_labels,
        tags=("quantum", "pauli", "labels", "exact"),
        discovery_terms=(
            "Pauli labels to exact vector",
            "construct Pauli from I X Y Z",
        ),
        examples=(
            OperationExample(
                name="single_qubit_y",
                description="Represent Y exactly as i X Z on the one-qubit register.",
                input={"register": _REGISTER, "labels": ["Y"], "phase": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="quantum.pauli.qubit.to_labels.compute",
        title="Express an exact qubit Pauli as labels",
        description=(
            "Return the exact local I/X/Y/Z row and its scalar phase under the "
            "same i^r X^x Z^z convention. The phase is relative to the labelled "
            "tensor product, so round trips preserve Y signs exactly."
        ),
        request_type=PauliToLabelsRequest,
        result_type=PauliToLabelsResult,
        run=_run_to_labels,
        tags=("quantum", "pauli", "labels", "exact"),
        discovery_terms=("exact Pauli to labels", "Pauli vector as I X Y Z"),
        examples=(
            OperationExample(
                name="negative_y",
                description="Return the labelled Y with its negative scalar phase.",
                input={
                    "pauli": {
                        "phase_free": {
                            "register": _REGISTER,
                            "x_bits": [1],
                            "z_bits": [1],
                        },
                        "phase": 3,
                    }
                },
            ),
        ),
    ),
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
        operation_id="quantum.pauli.family.commutation_matrix.compute",
        title="Compute a Pauli family's exact commutation matrix",
        description=(
            "Return the complete binary symplectic pairing matrix for an ordered "
            "named family of phase-free Paulis on one identical ordered register. "
            "The result retains the family as its row and column axis; diagonal "
            "entries are zero and the matrix is symmetric over GF(2)."
        ),
        request_type=PauliFamilyCommutationRequest,
        result_type=PauliFamilyCommutationResult,
        run=_run_family_commutation,
        tags=("quantum", "pauli", "commutation", "symplectic", "exact"),
        discovery_terms=("Pauli commutation matrix", "family symplectic pairings"),
        examples=(
            OperationExample(
                name="single_qubit_pauli_commutation_matrix",
                description="Compute pairings for X, Z, and Y on the same one-qubit register.",
                input={
                    "family": [
                        {"pauli_id": "x", "pauli": _X},
                        {"pauli_id": "z", "pauli": _Z},
                        {
                            "pauli_id": "y",
                            "pauli": {
                                "register": _REGISTER,
                                "x_bits": [1],
                                "z_bits": [1],
                            },
                        },
                    ]
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
        operation_id="quantum.stabilizer.syndrome.compute",
        title="Compute a Pauli error's stabilizer syndrome",
        description=(
            "Compute exact symplectic pairings against the canonical RREF basis "
            "of an isotropic binary check space. The result retains the check "
            "axis and error register; zero syndrome is equivalent to membership "
            "in the check space's symplectic orthogonal. Equal syndromes alone "
            "do not imply stabilizer equivalence."
        ),
        request_type=StabilizerSyndromeRequest,
        result_type=StabilizerSyndromeResult,
        run=_run_syndrome,
        tags=("quantum", "stabilizer", "syndrome", "symplectic", "exact"),
        examples=(
            OperationExample(
                name="bell_check_detects_local_x",
                description=(
                    "Compute the syndrome of X on the first qubit against the "
                    "ZZ check; the check space must be isotropic and share the "
                    "same ordered register."
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
                    },
                    "error": {
                        "register": {"qubit_ids": ["q0", "q1"]},
                        "x_bits": [1, 0],
                        "z_bits": [0, 0],
                    },
                },
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
