"""Finite class-function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.characters._models import (
    ClassFunctionInnerProductRequest,
    ClassFunctionInnerProductResult,
)
from jacobian.math.groups.characters.operations import class_function_inner_product


def _run_inner_product(
    request: ClassFunctionInnerProductRequest,
) -> ClassFunctionInnerProductResult:
    return class_function_inner_product(request.phi, request.psi)


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

TOOLS: tuple[MathTool[Any, Any], ...] = (
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
