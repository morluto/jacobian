"""Symmetric function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.symmetric_functions._models import (
    MAX_LR_SEARCH_STATES,
    MAX_LR_SKEW_CELLS,
    LittlewoodRichardsonCoefficientRequest,
    LittlewoodRichardsonCoefficientResult,
    SchurExpansionRequest,
    SchurExpansionResult,
    SchurProductRequest,
    SchurProductResult,
)
from jacobian.math.combinatorics.symmetric_functions.littlewood_richardson import (
    _compute_validated_lr,
    _schur_product_from_request,
)
from jacobian.math.combinatorics.symmetric_functions.operations import schur_evaluation


def _run_schur_evaluation(request: SchurExpansionRequest) -> SchurExpansionResult:
    return schur_evaluation(request.partition, request.point, request.variables)


def _run_littlewood_richardson_coefficient(
    request: LittlewoodRichardsonCoefficientRequest,
) -> LittlewoodRichardsonCoefficientResult:
    # The catalog parsed and admitted this request; run the shared
    # post-admission path without replaying request validation.
    return _compute_validated_lr(request)


def _run_schur_product(request: SchurProductRequest) -> SchurProductResult:
    return _schur_product_from_request(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="symmetric_function.schur.product.compute",
        title="Expand a bounded Schur product",
        description=(
            "Return the complete Schur-basis expansion of s_left times s_right "
            "using Littlewood-Richardson coefficients. The total degree is at "
            "most 8 and complete candidate-prefix work is pre-admitted."
        ),
        request_type=SchurProductRequest,
        result_type=SchurProductResult,
        run=_run_schur_product,
        tags=("symmetric-function", "schur", "littlewood-richardson", "exact"),
        discovery_terms=("Schur product", "Schur multiplication", "Schur expansion"),
        examples=(
            OperationExample(
                name="s2_times_s1",
                description="Expand s_(2) s_(1) = s_(3) + s_(2,1).",
                input={"left": {"parts": [2]}, "right": {"parts": [1]}},
            ),
        ),
    ),
    MathTool(
        operation_id="symmetric_function.schur.evaluate.compute",
        title="Evaluate a Schur function at a point",
        description="Evaluate the Schur function s_lambda(x_1,...,x_n) at a bounded "
        "integer point using the Jacobi-Trudi determinant formula.",
        request_type=SchurExpansionRequest,
        result_type=SchurExpansionResult,
        run=_run_schur_evaluation,
        tags=("symmetric-function", "schur", "exact"),
        examples=(
            OperationExample(
                name="schur_1_at_1_1",
                description="Evaluate s_(1)(1,1) = 2. Needs: decreasing positive parts "
                "with total size <=500 and at most 50 parts; distinct "
                "variables whose count equals the point length (both "
                "1..20); |coordinate| <=999999.",
                input={
                    "partition": {"parts": [1]},
                    "variables": ["x1", "x2"],
                    "point": [1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="symmetric_function.littlewood_richardson.coefficient.compute",
        title="Compute a Littlewood-Richardson coefficient",
        description=(
            "Compute c^outer_{inner, content} as the number of semistandard "
            "tableaux of skew shape outer/inner and content, using the "
            "right-to-left, top-to-bottom lattice-word convention. Complete "
            f"search is bounded to {MAX_LR_SKEW_CELLS} skew cells and "
            f"{MAX_LR_SEARCH_STATES} prefix states."
        ),
        request_type=LittlewoodRichardsonCoefficientRequest,
        result_type=LittlewoodRichardsonCoefficientResult,
        run=_run_littlewood_richardson_coefficient,
        tags=("symmetric-function", "schur", "littlewood-richardson", "exact"),
        discovery_terms=(
            "Littlewood-Richardson coefficient",
            "Schur product coefficient",
            "LR coefficient",
        ),
        examples=(
            OperationExample(
                name="coefficient_greater_than_one",
                description=(
                    "Compute c^(3,2,1)_(2,1),(2,1) under the fixed LR "
                    "reading-word convention."
                ),
                input={
                    "outer": {"parts": [3, 2, 1]},
                    "inner": {"parts": [2, 1]},
                    "content": {"parts": [2, 1]},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
