"""Finite class-function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.characters._models import (
    CharacterExteriorSquareRequest,
    CharacterKernel,
    CharacterKernelRequest,
    CharacterRingDecompositionRequest,
    CharacterRingDecompositionResult,
    CharacterRingElement,
    CharacterSymmetricSquareRequest,
    CharacterTableRequest,
    CharacterTableResult,
    CharacterTensorDecompositionRequest,
    CharacterTensorDecompositionResult,
    CharacterTensorProductRequest,
    ClassFunctionAddRequest,
    ClassFunctionConjugateRequest,
    ClassFunctionInductionRequest,
    ClassFunctionInductionResult,
    ClassFunctionInnerProductRequest,
    ClassFunctionInnerProductResult,
    ClassFunctionPointwiseProductRequest,
    ClassFunctionRestrictionRequest,
    ClassFunctionRestrictionResult,
    ClassFunctionScaleRequest,
    ClassPowerMapRequest,
    ClassPowerMapResult,
    CyclicCharacterRestrictionRequest,
    CyclicCharacterRestrictionResult,
    FiniteClassFunction,
    FrobeniusSchurIndicatorRequest,
    FrobeniusSchurIndicatorResult,
)
from jacobian.math.groups.characters.operations import (
    character_table,
    character_tensor_decomposition,
    class_function_add,
    class_function_conjugate,
    class_function_induce_from_subgroup,
    class_function_inner_product,
    class_function_pointwise_product,
    class_function_restrict_to_subgroup,
    class_function_scale,
    class_power_map,
    frobenius_schur_indicator,
    restrict_cyclic_character,
)
from jacobian.math.groups.characters.representation_ring_operations import (
    character_exterior_square,
    character_kernel,
    character_symmetric_square,
    character_tensor_product,
    class_function_character_decomposition,
)


def _run_inner_product(
    request: ClassFunctionInnerProductRequest,
) -> ClassFunctionInnerProductResult:
    return class_function_inner_product(request.phi, request.psi)


def _run_character_table(request: CharacterTableRequest) -> CharacterTableResult:
    return character_table(request.partition)


def _run_character_ring_decomposition(
    request: CharacterRingDecompositionRequest,
) -> CharacterRingDecompositionResult:
    return class_function_character_decomposition(request)


def _run_character_tensor_product(
    request: CharacterTensorProductRequest,
) -> CharacterRingElement:
    return character_tensor_product(request)


def _run_character_symmetric_square(
    request: CharacterSymmetricSquareRequest,
) -> CharacterRingElement:
    return character_symmetric_square(request)


def _run_character_exterior_square(
    request: CharacterExteriorSquareRequest,
) -> CharacterRingElement:
    return character_exterior_square(request)


def _run_character_kernel(request: CharacterKernelRequest) -> CharacterKernel:
    return character_kernel(request)


def _run_tensor_decomposition(
    request: CharacterTensorDecompositionRequest,
) -> CharacterTensorDecompositionResult:
    return character_tensor_decomposition(request)


def _run_pointwise_product(
    request: ClassFunctionPointwiseProductRequest,
) -> FiniteClassFunction:
    return class_function_pointwise_product(request.phi, request.psi)


def _run_scale(request: ClassFunctionScaleRequest) -> FiniteClassFunction:
    return class_function_scale(request.function, request.scalar)


def _run_add(request: ClassFunctionAddRequest) -> FiniteClassFunction:
    return class_function_add(request.phi, request.psi)


def _run_cyclic_character_restriction(
    request: CyclicCharacterRestrictionRequest,
) -> CyclicCharacterRestrictionResult:
    return restrict_cyclic_character(request)


def _run_conjugate(
    request: ClassFunctionConjugateRequest,
) -> FiniteClassFunction:
    return class_function_conjugate(request.function)


def _c2_trivial_class_function() -> dict[str, Any]:
    return {
        "axis": {
            "class_sizes": [1, 1],
            "group_order": 2,
            "cyclotomic_order": 1,
            "group": {"degree": 3, "generators": [[1, 0, 2]]},
            "class_representatives": [[0, 1, 2], [1, 0, 2]],
        },
        "values": [_rational_value(1), _rational_value(1)],
    }


def _character_partition(
    group: dict[str, Any], classes: list[list[list[int]]]
) -> dict[str, Any]:
    return {"source": group, "classes": classes}


def _rational_value(numerator: int) -> dict[str, Any]:
    return {
        "order": 1,
        "coefficients": [{"num": str(numerator), "den": "1"}],
    }


def _cyclotomic_value(order: int, coefficients: list[int]) -> dict[str, Any]:
    return {
        "order": order,
        "coefficients": [
            {"num": str(coefficient), "den": "1"} for coefficient in coefficients
        ],
    }


def _s3_class_axis() -> dict[str, Any]:
    return {
        "class_sizes": [1, 3, 2],
        "group_order": 6,
        "cyclotomic_order": 1,
    }


def _s3_standard_class_function() -> dict[str, Any]:
    return {
        "axis": {
            "class_sizes": [1, 3, 2],
            "group_order": 6,
            "cyclotomic_order": 1,
            "group": {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
            "class_representatives": [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, 0],
            ],
        },
        "values": [_rational_value(2), _rational_value(0), _rational_value(-1)],
    }


def _s3_character_table() -> dict[str, Any]:
    def _value(numerator: int) -> dict[str, Any]:
        return _cyclotomic_value(6, [numerator, 0])

    return {
        "partition": _S3_PARTITION,
        "axis": {
            "class_sizes": [1, 3, 2],
            "group_order": 6,
            "cyclotomic_order": 6,
            "group": {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
            "class_representatives": [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, 0],
            ],
        },
        "rows": [
            {
                "label": "trivial",
                "degree": 1,
                "values": [_value(1), _value(1), _value(1)],
            },
            {
                "label": "sign",
                "degree": 1,
                "values": [_value(1), _value(-1), _value(1)],
            },
            {
                "label": "standard",
                "degree": 2,
                "values": [
                    _value(2),
                    _value(0),
                    _value(-1),
                ],
            },
        ],
        "degree_square_sum": 6,
    }


def _s3_ring_element(coordinates: list[int]) -> dict[str, Any]:
    return {
        "table": _s3_character_table(),
        "irreducible_multiplicities": coordinates,
    }


_S3_TRIVIAL = {
    "axis": _s3_class_axis(),
    "values": [
        _rational_value(1),
        _rational_value(1),
        _rational_value(1),
    ],
}

_S3_SIGN = {
    "axis": _s3_class_axis(),
    "values": [
        _rational_value(1),
        _rational_value(-1),
        _rational_value(1),
    ],
}

_C3_LINEAR_CHARACTER = {
    "axis": {
        "class_sizes": [1, 1, 1],
        "group_order": 3,
        "cyclotomic_order": 3,
    },
    "values": [
        _cyclotomic_value(3, [1, 0]),
        _cyclotomic_value(3, [0, 1]),
        _cyclotomic_value(3, [-1, -1]),
    ],
}

_TRIVIAL_PARTITION = _character_partition({"degree": 1, "generators": [[0]]}, [[[0]]])
_S3_PARTITION = _character_partition(
    {"degree": 3, "generators": [[1, 2, 0], [1, 0, 2]]},
    [
        [[0, 1, 2]],
        [[0, 2, 1], [1, 0, 2], [2, 1, 0]],
        [[1, 2, 0], [2, 0, 1]],
    ],
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="finite_group.cyclic_character.restrict.compute",
        title="Restrict a cyclic-group irreducible character",
        description=(
            "Restrict one row of the exact complete character table of a supported "
            "cyclic permutation group C_n to its unique subgroup C_d, where d divides "
            "n. The result retains the source table, embedded target subgroup, and "
            "target-to-source class map. This returns restricted values, not a "
            "decomposition into target irreducibles."
        ),
        request_type=CyclicCharacterRestrictionRequest,
        result_type=CyclicCharacterRestrictionResult,
        run=_run_cyclic_character_restriction,
        tags=("finite-group", "character", "restriction", "cyclic", "exact"),
        discovery_terms=(
            "restrict cyclic irreducible character",
            "cyclic subgroup character restriction",
        ),
        examples=(
            OperationExample(
                name="c4_character_restricted_to_c2",
                description="Restrict the second Fourier character of C4 to its unique subgroup C2.",
                input={
                    "partition": _character_partition(
                        {"degree": 4, "generators": [[1, 2, 3, 0]]},
                        [
                            [[0, 1, 2, 3]],
                            [[1, 2, 3, 0]],
                            [[2, 3, 0, 1]],
                            [[3, 0, 1, 2]],
                        ],
                    ),
                    "row_index": 1,
                    "subgroup_order": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="group.class_power_map.compute",
        title="Compute a finite-group conjugacy-class power map",
        description=(
            "For a complete canonical conjugacy partition of a concrete finite "
            "permutation group, return the class index containing g^k for each "
            "source class. The source partition is re-established; group order "
            "is bounded by 256 and k by 1,000,000."
        ),
        request_type=ClassPowerMapRequest,
        result_type=ClassPowerMapResult,
        run=class_power_map,
        tags=("group", "character", "conjugacy", "power-map", "exact"),
        discovery_terms=("class power map", "conjugacy class k-th powers"),
        examples=(
            OperationExample(
                name="s3_square_power_map",
                description="Square each conjugacy class of S3 and return its image class.",
                input={"partition": _S3_PARTITION, "exponent": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="group.character.frobenius_schur_indicator.compute",
        title="Compute an ordinary second Frobenius-Schur indicator",
        description=(
            "For one irreducible row of a complete exact ordinary character table "
            "of a supported concrete finite group, compute nu_2(chi) = "
            "|G|^-1 sum_C |C| chi(c^2) using the class-square map. Returns the "
            "integer -1, 0, or 1 and retains the exact source table and row index. "
            "This operation covers ordinary second indicators only, not modular "
            "or higher indicators. GAP's character-table manual gives the general "
            "ordinary formula nu_n(chi) = |G|^-1 sum_g chi(g^n)."
        ),
        request_type=FrobeniusSchurIndicatorRequest,
        result_type=FrobeniusSchurIndicatorResult,
        run=frobenius_schur_indicator,
        tags=("group", "character", "frobenius-schur", "exact"),
        discovery_terms=(
            "second Frobenius Schur indicator",
            "real or quaternionic irreducible",
        ),
        examples=(
            OperationExample(
                name="s3_standard_real_indicator",
                description="The standard irreducible character of S3 has indicator 1.",
                input={"table": _s3_character_table(), "row_index": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="class_function.conjugate.compute",
        title="Conjugate an exact finite-group class function",
        description=(
            "Apply exact complex conjugation valuewise using the cyclotomic "
            "involution zeta -> zeta^-1. The exact class axis is retained; "
            "the operation accepts general class functions and makes no "
            "character or irreducibility claim. Work, coefficient growth, "
            "and output size are bounded before cyclotomic expansion."
        ),
        request_type=ClassFunctionConjugateRequest,
        result_type=FiniteClassFunction,
        run=_run_conjugate,
        tags=("group", "character", "class-function", "exact", "conjugation"),
        discovery_terms=(
            "complex conjugate class function",
            "Galois conjugate finite group class function",
            "cyclotomic conjugation",
        ),
        examples=(
            OperationExample(
                name="conjugate_c3_linear_character",
                description=(
                    "Conjugate a linear character of C3 by inverting each "
                    "root-of-unity value."
                ),
                input={"function": _C3_LINEAR_CHARACTER},
            ),
        ),
    ),
    MathTool(
        operation_id="class_function.add.compute",
        title="Add exact finite-group class functions",
        description=(
            "Add exact cyclotomic values coordinatewise on an identical class "
            "axis. The output retains that axis and accepts arbitrary class "
            "functions. Input size, arithmetic work, coefficient growth, and "
            "serialized output are bounded before exact addition."
        ),
        request_type=ClassFunctionAddRequest,
        result_type=FiniteClassFunction,
        run=_run_add,
        tags=("group", "character", "class-function", "addition", "exact"),
        discovery_terms=(
            "add finite group class functions",
            "sum of class functions",
            "class function linear combination",
        ),
        examples=(
            OperationExample(
                name="s3_trivial_plus_standard",
                description=(
                    "Add the trivial and standard characters of S3 on the same "
                    "class axis, yielding values (3, 1, 0)."
                ),
                input={
                    "phi": {
                        "axis": _s3_class_axis(),
                        "values": [_rational_value(1)] * 3,
                    },
                    "psi": {
                        "axis": _s3_class_axis(),
                        "values": [
                            _rational_value(2),
                            _rational_value(0),
                            _rational_value(-1),
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="class_function.pointwise_multiply.compute",
        title="Multiply exact finite-group class functions pointwise",
        description=(
            "Multiply exact cyclotomic values class by class on an identical "
            "class axis. The output retains that axis and is the character of "
            "the tensor product when both inputs are characters. Exact work, "
            "intermediate coefficients, and output cells are bounded before "
            "cyclotomic arithmetic."
        ),
        request_type=ClassFunctionPointwiseProductRequest,
        result_type=FiniteClassFunction,
        run=_run_pointwise_product,
        tags=("group", "character", "class-function", "tensor-product", "exact"),
        discovery_terms=(
            "class function pointwise product",
            "tensor product character",
            "product of finite group characters",
        ),
        examples=(
            OperationExample(
                name="s3_trivial_times_standard",
                description=(
                    "Pointwise multiplication by the trivial S3 character returns "
                    "the standard character on the same class axis."
                ),
                input={
                    "phi": {
                        "axis": _s3_class_axis(),
                        "values": [_rational_value(1)] * 3,
                    },
                    "psi": {
                        "axis": _s3_class_axis(),
                        "values": [
                            _rational_value(2),
                            _rational_value(0),
                            _rational_value(-1),
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="class_function.scale.compute",
        title="Scale an exact finite-group class function",
        description=(
            "Multiply every value of an exact class function by one scalar in "
            "the cyclotomic field named by its class axis. The axis is retained; "
            "the result is an ordinary class function, with no character claim. "
            "Cyclotomic work, coefficient growth, and output bytes use the "
            "bounded pointwise-product admission before arithmetic."
        ),
        request_type=ClassFunctionScaleRequest,
        result_type=FiniteClassFunction,
        run=_run_scale,
        tags=("group", "character", "class-function", "scalar-action", "exact"),
        discovery_terms=(
            "scale finite group class function",
            "scalar multiple of a class function",
            "character linear combination scalar",
        ),
        examples=(
            OperationExample(
                name="scale_s3_standard_by_minus_two",
                description="Scale the standard S3 character by -2 on its unchanged class axis.",
                input={
                    "scalar": _rational_value(-2),
                    "function": _s3_standard_class_function(),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_group.character.tensor_product.decompose.compute",
        title="Decompose a tensor product of canonical S3 characters",
        description=(
            "Take the pointwise tensor product of two irreducible rows in the "
            "canonical complete S3 table and return its exact multiplicities in "
            "that basis. The supported group is currently S3."
        ),
        request_type=CharacterTensorDecompositionRequest,
        result_type=CharacterTensorDecompositionResult,
        run=_run_tensor_decomposition,
        tags=("finite-group", "character", "tensor-product", "decomposition", "exact"),
        discovery_terms=(
            "decompose tensor product of irreducible characters",
            "S3 character tensor decomposition",
        ),
        examples=(
            OperationExample(
                name="s3_standard_tensor_square",
                description=(
                    "Decompose standard tensor standard as trivial plus sign plus "
                    "standard for S3."
                ),
                input={
                    "partition": _S3_PARTITION,
                    "left_row_index": 2,
                    "right_row_index": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_group.character_table.compute",
        title="Compute a complete character table for a bounded finite group",
        description=(
            "Compute the complete exact irreducible character table of a concrete "
            "permutation group class partition. This bounded release supports the "
            "trivial, cyclic, and S3 groups; the returned rows retain the complete "
            "source group and ordered class partition, and satisfy row orthogonality "
            "and the degree-square identity."
        ),
        request_type=CharacterTableRequest,
        result_type=CharacterTableResult,
        run=_run_character_table,
        tags=("group", "character", "character-table", "exact"),
        discovery_terms=(
            "finite group character table",
            "irreducible character rows",
            "character orthogonality",
        ),
        examples=(
            OperationExample(
                name="s3_character_table",
                description=(
                    "Compute the complete three-row character table of S3; the "
                    "partition must be a complete canonical conjugacy-class partition."
                ),
                input={"partition": _S3_PARTITION},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_group.class_function.character_ring_decompose.compute",
        title="Express a class function in a bounded character basis",
        description=(
            "For a class function on a supported concrete finite group, compute "
            "its exact integer coordinates in the canonical irreducible-character "
            "basis and return a table-bound virtual-character ring element. "
            "Signed coordinates are allowed; nonnegative coordinates are ordinary "
            "characters. Supported groups currently are the trivial, cyclic, and S3 groups."
        ),
        request_type=CharacterRingDecompositionRequest,
        result_type=CharacterRingDecompositionResult,
        run=_run_character_ring_decomposition,
        tags=(
            "finite-group",
            "character",
            "decomposition",
            "representation-ring",
            "exact",
        ),
        discovery_terms=(
            "decompose finite group class function",
            "virtual character irreducible multiplicities",
            "finite group representation ring coordinates",
        ),
        examples=(
            OperationExample(
                name="s3_standard_character_coordinates",
                description=(
                    "Express the standard S3 character in the complete irreducible basis."
                ),
                input={"class_function": _s3_standard_class_function()},
            ),
        ),
    ),
    MathTool(
        operation_id="character.tensor_product.compute",
        title="Decompose a bounded virtual-character tensor product",
        description=(
            "Multiply two table-bound virtual characters pointwise and return "
            "their exact irreducible multiplicities. Inputs must use the same "
            "canonical character table. Supported groups currently are the "
            "trivial group, cyclic groups through order 60, and S3. Signed "
            "coordinates are preserved for virtual characters."
        ),
        request_type=CharacterTensorProductRequest,
        result_type=CharacterRingElement,
        run=_run_character_tensor_product,
        tags=(
            "finite-group",
            "character",
            "tensor-product",
            "representation-ring",
            "exact",
        ),
        discovery_terms=(
            "tensor product virtual characters decompose irreducible multiplicities",
            "finite group representation ring product",
            "character table tensor product decomposition",
        ),
        examples=(
            OperationExample(
                name="s3_standard_tensor_square",
                description=(
                    "The standard two-dimensional S3 representation tensored "
                    "with itself decomposes as trivial plus sign plus standard."
                ),
                input={
                    "left": _s3_ring_element([0, 0, 1]),
                    "right": _s3_ring_element([0, 0, 1]),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="character.symmetric_square.compute",
        title="Compute a finite-group character's second symmetric power",
        description=(
            "Apply Sym^2(x) = (x tensor x + psi^2(x))/2 to one exact, "
            "table-bound virtual character. The canonical table is rebuilt "
            "from its concrete group, and the class squaring map is derived "
            "from that complete partition. Supported groups are trivial, "
            "cyclic of order at most 60, and S3; signed virtual coordinates "
            "are accepted when the result has exact integral coordinates."
        ),
        request_type=CharacterSymmetricSquareRequest,
        result_type=CharacterRingElement,
        run=_run_character_symmetric_square,
        tags=("finite-group", "character", "symmetric-power", "exact"),
        discovery_terms=(
            "symmetric square of a finite group character",
            "second symmetric power virtual character",
            "symmetric square representation ring",
        ),
        examples=(
            OperationExample(
                name="s3_standard_symmetric_square",
                description="Sym^2 of the standard S3 representation is 1 plus standard.",
                input={"character": _s3_ring_element([0, 0, 1])},
            ),
        ),
    ),
    MathTool(
        operation_id="character.exterior_square.compute",
        title="Compute a finite-group character's second exterior power",
        description=(
            "Apply Lambda^2(x) = (x tensor x - psi^2(x))/2 to one exact, "
            "table-bound virtual character. The canonical table is rebuilt "
            "from its concrete group, and the class squaring map is derived "
            "from that complete partition. Supported groups are trivial, "
            "cyclic of order at most 60, and S3; signed virtual coordinates "
            "are accepted when the result has exact integral coordinates."
        ),
        request_type=CharacterExteriorSquareRequest,
        result_type=CharacterRingElement,
        run=_run_character_exterior_square,
        tags=("finite-group", "character", "exterior-power", "exact"),
        discovery_terms=(
            "exterior square of a finite group character",
            "second exterior power virtual character",
            "exterior square representation ring",
        ),
        examples=(
            OperationExample(
                name="s3_standard_exterior_square",
                description="Lambda^2 of the standard S3 representation is sign.",
                input={"character": _s3_ring_element([0, 0, 1])},
            ),
        ),
    ),
    MathTool(
        operation_id="character.kernel.compute",
        title="Compute the kernel of a bounded ordinary character",
        description=(
            "Return the subgroup on which an exact ordinary character has "
            "value equal to its degree. Input coordinates must be nonnegative "
            "in the canonical irreducible basis, and the canonical character "
            "table is rebuilt from its concrete group. The exact cyclotomic "
            "trace equality is equivalent to identity action. Supported "
            "groups are trivial, S3, and cyclic groups of order n satisfying "
            "n^2*phi(n) <= 100000, the canonical table cell bound. The "
            "result retains both the ambient group and its kernel subgroup."
        ),
        request_type=CharacterKernelRequest,
        result_type=CharacterKernel,
        run=_run_character_kernel,
        tags=("finite-group", "character", "kernel", "exact"),
        discovery_terms=(
            "kernel subgroup of a finite group character",
            "normal subgroup where a representation acts trivially",
            "character kernel from exact irreducible multiplicities",
        ),
        examples=(
            OperationExample(
                name="s3_sign_character_kernel",
                description="The sign character of S3 has kernel A3.",
                input={"character": _s3_ring_element([0, 1, 0])},
            ),
        ),
    ),
    MathTool(
        operation_id="class_function.inner_product.compute",
        title="Compute the exact Hermitian inner product of two class functions",
        description=(
            "Compute the standard Hermitian inner product "
            "<phi,psi> = (1/|G|) sum_C |C| phi(C) conjugate(psi(C)) of two "
            "exact class functions on one shared conjugacy-class axis. Both "
            "requests must carry the identical class sizes, group order, and "
            "cyclotomic order; values are exact rational or root-of-unity "
            "cyclotomic elements. The result includes the complete per-class "
            "contribution table and the exact inner product."
        ),
        request_type=ClassFunctionInnerProductRequest,
        result_type=ClassFunctionInnerProductResult,
        run=_run_inner_product,
        tags=("group", "character", "class-function", "inner-product", "exact"),
        discovery_terms=(
            "class function inner product",
            "character orthogonality",
            "Hermitian inner product of characters",
            "exact character pairing",
        ),
        examples=(
            OperationExample(
                name="s3_trivial_sign_inner_product",
                description=(
                    "Pair the trivial and sign class functions of S3 on the "
                    "three classes of sizes 1, 3, 2. The two class functions "
                    "must share the exact class axis; the result is <triv,sign> "
                    "= 0 with the complete contribution table."
                ),
                input={"phi": _S3_TRIVIAL, "psi": _S3_SIGN},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_group.class_function.restrict_to_subgroup.compute",
        title="Restrict a finite-group class function to a subgroup",
        description=(
            "Restrict an exact cyclotomic class function along an explicit "
            "same-domain permutation subgroup. Every subgroup generator must "
            "belong to the concrete source group. The result includes the "
            "target-class-to-source-class map and restricted exact values. "
            "Both group orders are at most 256; the target order is at most 128."
        ),
        request_type=ClassFunctionRestrictionRequest,
        result_type=ClassFunctionRestrictionResult,
        run=class_function_restrict_to_subgroup,
        tags=("group", "character", "class-function", "restriction", "exact"),
        examples=(
            OperationExample(
                name="s3_standard_restricted_to_transposition_subgroup",
                description=(
                    "Restrict the standard class function of S3 to the subgroup "
                    "generated by a transposition, obtaining values (2, 0)."
                ),
                input={
                    "class_function": _s3_standard_class_function(),
                    "subgroup": {"degree": 3, "generators": [[1, 0, 2]]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_group.class_function.induce.compute",
        title="Induce a finite-group class function from a subgroup",
        description=(
            "Compute the exact induced class function using "
            "Ind_H^G(phi)(g) = (1/|H|) sum_x phi(x^-1*g*x) over parent "
            "elements whose conjugate lies in H. The class-function axis "
            "retains H; the parent is an explicit same-domain permutation "
            "group and every subgroup generator must belong to it. The result "
            "retains both canonical class partitions, the subgroup-class to "
            "parent-class map, and exact cyclotomic values. Subgroup order is "
            "at most 128, parent order at most 256, and work, coefficient "
            "height, and output bytes are admitted before conjugacy expansion."
        ),
        request_type=ClassFunctionInductionRequest,
        result_type=ClassFunctionInductionResult,
        run=class_function_induce_from_subgroup,
        tags=("group", "character", "class-function", "induction", "exact"),
        discovery_terms=(
            "induce finite group character from subgroup",
            "class function induction",
            "induced character exact values",
            "Frobenius reciprocity class functions",
        ),
        examples=(
            OperationExample(
                name="s3_induce_trivial_from_transposition_subgroup",
                description=(
                    "Induce the trivial class function of a transposition C2 "
                    "subgroup to S3. The subgroup-class map is (identity, "
                    "transposition) -> (identity, transposition), and the "
                    "induced values on (identity, transposition, 3-cycle) are (3, 1, 0)."
                ),
                input={
                    "class_function": _c2_trivial_class_function(),
                    "parent_group": {
                        "degree": 3,
                        "generators": [[1, 2, 0], [1, 0, 2]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
