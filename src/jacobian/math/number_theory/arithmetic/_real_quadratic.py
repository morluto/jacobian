"""Typed real-quadratic order operation and checker declaration."""

from __future__ import annotations

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.algebraic_numbers.quadratic import (
    RealQuadraticEmbeddingProfile,
    RealQuadraticOrderValue,
    RealQuadraticValue,
    real_quadratic_embeddings,
    real_quadratic_order,
)


class RealQuadraticOrderRequest(StrictModel):
    """One bounded comparison in a shared real quadratic field."""

    left: RealQuadraticValue
    right: RealQuadraticValue


class RealQuadraticEmbeddingRequest(StrictModel):
    """One bounded element whose two real embeddings should be materialized."""

    element: RealQuadraticValue


def _compute_real_quadratic_order(
    request: RealQuadraticOrderRequest,
) -> RealQuadraticOrderValue:
    """Project the wire request onto the native canonical-value kernel."""

    return real_quadratic_order(request.left, request.right)


def _compute_real_quadratic_embeddings(
    request: RealQuadraticEmbeddingRequest,
) -> RealQuadraticEmbeddingProfile:
    """Project the wire request onto the exact embedding kernel."""

    return real_quadratic_embeddings(request.element)


REAL_QUADRATIC_OPERATIONS = (
    MathTool(
        operation_id="arithmetic.real_quadratic.order.compute",
        title="Compare exact real quadratic values",
        description=(
            "Compare two bounded values a+b*sqrt(d) in one shared real quadratic "
            "field, returning their exact difference and squared-magnitude sign data."
        ),
        request_type=RealQuadraticOrderRequest,
        result_type=RealQuadraticOrderValue,
        run=_compute_real_quadratic_order,
        tags=("arithmetic", "real-quadratic", "quadratic-surd", "exact-order"),
        examples=(
            OperationExample(
                name="pang_m4_scalar_gap",
                description="Compare 3*sqrt(3)/8 with 1/2+sqrt(3)/20 exactly.",
                input={
                    "left": {
                        "rational_part": {"num": "0", "den": "1"},
                        "radical_coefficient": {"num": "3", "den": "8"},
                        "radicand": 3,
                    },
                    "right": {
                        "rational_part": {"num": "1", "den": "2"},
                        "radical_coefficient": {"num": "1", "den": "20"},
                        "radicand": 3,
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic.real_quadratic.embeddings.compute",
        title="Compute both real quadratic embeddings",
        description=(
            "Return the exact positive-root and conjugate-root embeddings, "
            "trace, and norm of one bounded element a+b*sqrt(d)."
        ),
        request_type=RealQuadraticEmbeddingRequest,
        result_type=RealQuadraticEmbeddingProfile,
        run=_compute_real_quadratic_embeddings,
        tags=("arithmetic", "real-quadratic", "embeddings", "exact"),
        examples=(
            OperationExample(
                name="sqrt_2_embedding_profile",
                description="Compute both embeddings of 1 + sqrt(2).",
                input={
                    "element": {
                        "rational_part": {"num": "1", "den": "1"},
                        "radical_coefficient": {"num": "1", "den": "1"},
                        "radicand": 2,
                    }
                },
            ),
        ),
    ),
)

__all__ = ["REAL_QUADRATIC_OPERATIONS", "RealQuadraticEmbeddingRequest"]
