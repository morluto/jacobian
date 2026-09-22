"""Finite class-function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.characters._models import (
    CharacterTableRequest,
    CharacterTableResult,
    ClassFunctionInnerProductRequest,
    ClassFunctionInnerProductResult,
)
from jacobian.math.groups.characters.operations import (
    character_table,
    class_function_inner_product,
)


def _run_inner_product(
    request: ClassFunctionInnerProductRequest,
) -> ClassFunctionInnerProductResult:
    return class_function_inner_product(request.phi, request.psi)


def _run_character_table(request: CharacterTableRequest) -> CharacterTableResult:
    return character_table(request.partition)


def _character_partition(
    group: dict[str, Any], classes: list[list[list[int]]]
) -> dict[str, Any]:
    return {"source": group, "classes": classes}


def _rational_value(numerator: int) -> dict[str, Any]:
    return {
        "order": 1,
        "coefficients": [{"num": str(numerator), "den": "1"}],
    }


def _s3_class_axis() -> dict[str, Any]:
    return {
        "class_sizes": [1, 3, 2],
        "group_order": 6,
        "cyclotomic_order": 1,
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
)

__all__ = ["TOOLS"]
