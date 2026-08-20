"""Plane algebraic curve operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.plane_algebraic_curves._models import (
    AffineChartRequest,
    AffineChartResult,
    AffineCurveRequest,
    AffineCurveResult,
    ProjectiveClosureRequest,
    ProjectiveClosureResult,
)
from jacobian.math.plane_algebraic_curves._operations import (
    compute_affine_chart,
    compute_affine_curve_check,
    compute_projective_closure,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "algebraic_geometry.affine_plane_curve.check",
        "Check an affine plane curve",
        "Check that a polynomial defines a valid affine plane curve f(x,y)=0 "
        "and return its degree.",
        AffineCurveRequest,
        AffineCurveResult,
        compute_affine_curve_check,
        "algebraic-geometry",
        "affine-curve",
        "exact",
        examples=(
            example(
                "circle",
                "Check the unit circle x^2 + y^2 - 1 = 0.",
                {
                    "variables": ["x", "y"],
                    "polynomial": "x**2 + y**2 - 1",
                },
            ),
        ),
    ),
    _op(
        "algebraic_geometry.plane_curve.projective_closure.compute",
        "Compute the projective closure of an affine curve",
        "Homogenize an affine plane curve to obtain its projective closure.",
        ProjectiveClosureRequest,
        ProjectiveClosureResult,
        compute_projective_closure,
        "algebraic-geometry",
        "projective-closure",
        "exact",
        examples=(
            example(
                "circle_closure",
                "Projective closure of x^2 + y^2 - 1.",
                {
                    "variables": ["x", "y"],
                    "polynomial": "x**2 + y**2 - 1",
                },
            ),
        ),
    ),
    _op(
        "algebraic_geometry.projective_curve.affine_chart.compute",
        "Extract an affine chart from a projective curve",
        "Dehomogenize a projective curve at the given chart variable by "
        "setting that variable to 1.",
        AffineChartRequest,
        AffineChartResult,
        compute_affine_chart,
        "algebraic-geometry",
        "affine-chart",
        "exact",
        examples=(
            example(
                "chart_z",
                "Extract the z=1 chart of x^2 + y^2 - z^2.",
                {
                    "variables": ["x", "y", "z"],
                    "polynomial": "x**2 + y**2 - z**2",
                    "chart_variable": "z",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
