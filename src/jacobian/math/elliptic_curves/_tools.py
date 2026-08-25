"""Typed declarations for elliptic curve operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.elliptic_curves._models import (
    CurveDiscriminantResult,
    CurvePointRequest,
    EllipticCurvePointAdditionRequest,
    EllipticCurvePointResult,
    EllipticCurveRequest,
    PointOnCurveResult,
    ScalarMultiplicationRequest,
    ScalarMultiplicationResult,
)
from jacobian.math.elliptic_curves._operations import (
    add_points,
    check_point_on_curve,
    compute_discriminant,
    scalar_multiply,
)


def elliptic_curve_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_DISCRIMINANT_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
}

_POINT_ON_CURVE_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
    "point": {
        "x": {"num": "0", "den": "1"},
        "y": {"num": "0", "den": "1"},
    },
}

_POINT_ADDITION_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
    "first": {
        "curve": {
            "coefficient_a": {"num": "1", "den": "1"},
            "coefficient_b": {"num": "0", "den": "1"},
        },
        "point": {
            "x": {"num": "0", "den": "1"},
            "y": {"num": "0", "den": "1"},
        },
        "at_infinity": False,
    },
    "second": {
        "curve": {
            "coefficient_a": {"num": "1", "den": "1"},
            "coefficient_b": {"num": "0", "den": "1"},
        },
        "point": {
            "x": {"num": "0", "den": "1"},
            "y": {"num": "0", "den": "1"},
        },
        "at_infinity": False,
    },
}

_SCALAR_MULT_EXAMPLE: dict[str, Any] = {
    "curve": {
        "coefficient_a": {"num": "-1", "den": "1"},
        "coefficient_b": {"num": "0", "den": "1"},
    },
    "point": {
        "curve": {
            "coefficient_a": {"num": "-1", "den": "1"},
            "coefficient_b": {"num": "0", "den": "1"},
        },
        "point": {
            "x": {"num": "1", "den": "1"},
            "y": {"num": "0", "den": "1"},
        },
        "at_infinity": False,
    },
    "scalar": 2,
}


ELLIPTIC_CURVE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    elliptic_curve_operation(
        "number_theory.elliptic_curve.short_weierstrass.discriminant.compute",
        "Compute the discriminant of a short Weierstrass elliptic curve",
        "Compute the exact discriminant Δ = -16(4A³ + 27B²) of a short "
        "Weierstrass curve y² = x³ + Ax + B over QQ, together with the "
        "nonsingularity predicate (Δ ≠ 0).",
        EllipticCurveRequest,
        CurveDiscriminantResult,
        compute_discriminant,
        "elliptic-curve",
        "discriminant",
        "exact",
        examples=(
            example(
                "y_squared_equals_x_cubed_plus_x",
                "Compute the discriminant of y² = x³ + x (A=1, B=0).",
                _DISCRIMINANT_EXAMPLE,
            ),
        ),
    ),
    elliptic_curve_operation(
        "number_theory.elliptic_curve.short_weierstrass.point_on_curve.decide",
        "Check whether a point lies on a short Weierstrass elliptic curve",
        "Check whether a rational affine point (x, y) lies on the curve "
        "y² = x³ + Ax + B by verifying y² = x³ + Ax + B exactly over QQ.",
        CurvePointRequest,
        PointOnCurveResult,
        check_point_on_curve,
        "elliptic-curve",
        "point-on-curve",
        "exact",
        examples=(
            example(
                "origin_on_x_cubed_plus_x",
                "Check whether (0, 0) lies on y² = x³ + x.",
                _POINT_ON_CURVE_EXAMPLE,
            ),
        ),
    ),
    elliptic_curve_operation(
        "number_theory.elliptic_curve.short_weierstrass.point_addition.compute",
        "Add two points on a short Weierstrass elliptic curve",
        "Add two rational affine points P₁ + P₂ on y² = x³ + Ax + B using "
        "the exact chord-and-tangent group law over QQ. Returns the point "
        "at infinity when P₁ + P₂ = O.",
        EllipticCurvePointAdditionRequest,
        EllipticCurvePointResult,
        add_points,
        "elliptic-curve",
        "point-addition",
        "group-law",
        "exact",
        examples=(
            example(
                "double_origin_on_x_cubed_plus_x",
                "Compute (0,0) + (0,0) on y² = x³ + x; the result is at infinity.",
                _POINT_ADDITION_EXAMPLE,
            ),
        ),
    ),
    elliptic_curve_operation(
        "number_theory.elliptic_curve.short_weierstrass.scalar_multiply.compute",
        "Compute n*P on a short Weierstrass elliptic curve",
        "Compute the scalar multiple n*P on y² = x³ + Ax + B using the "
        "double-and-add method over QQ. Returns the point at infinity "
        "when n*P = O.",
        ScalarMultiplicationRequest,
        ScalarMultiplicationResult,
        scalar_multiply,
        "elliptic-curve",
        "scalar-multiplication",
        "group-law",
        "exact",
        examples=(
            example(
                "double_point_on_x_cubed_minus_x",
                "Compute 2*(1,0) on y² = x³ - x; the result is at infinity.",
                _SCALAR_MULT_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = ELLIPTIC_CURVE_OPERATIONS

__all__ = ["TOOLS"]
