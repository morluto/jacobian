"""Plane algebraic curve operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.algebraic_curves._models import (
    AffineChartRequest,
    AffineChartResult,
    AffineCurveRequest,
    AffineCurveResult,
    ProjectiveClosureRequest,
    ProjectiveClosureResult,
    RationalConicParametrizationRequest,
    RationalConicParametrizationResult,
)
from jacobian.math.geometry.algebraic_curves.operations import (
    affine_chart,
    affine_curve_check,
    projective_closure,
    rational_conic_parametrization,
)


def compute_affine_curve_check(request: AffineCurveRequest) -> AffineCurveResult:
    """Unpack one request and project the native curve check to its wire result."""
    is_valid, degree = affine_curve_check(request.polynomial)
    return AffineCurveResult(is_valid=is_valid, degree=degree)


def compute_projective_closure(
    request: ProjectiveClosureRequest,
) -> ProjectiveClosureResult:
    """Unpack one request and project the native closure to its wire result."""
    return ProjectiveClosureResult(
        polynomial=projective_closure(request.polynomial),
    )


def compute_affine_chart(request: AffineChartRequest) -> AffineChartResult:
    """Unpack one request and project the native chart to its wire result."""
    return AffineChartResult(
        polynomial=affine_chart(request.polynomial, request.chart_variable),
    )


def compute_rational_conic_parametrization(
    request: RationalConicParametrizationRequest,
) -> RationalConicParametrizationResult:
    """Unpack one request and project the native conic data to its wire result."""
    data = rational_conic_parametrization(
        request.polynomial,
        request.point,
        request.parameter,
    )
    return RationalConicParametrizationResult._from_kernel(
        request,
        coordinates=data.coordinates,
        inverse_parameter=data.inverse_parameter,
        finite_parameter_denominator=data.finite_parameter_denominator,
    )


def _polynomial(
    variables: tuple[str, ...],
    *terms: tuple[int, tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": list(variables),
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": list(exponents),
                }
                for coefficient, exponents in terms
            ]
        },
    }


def _op[RequestT: StrictModel, ResultT: StrictModel](
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
                "Check the unit circle x^2 + y^2 - 1 = 0; an affine plane "
                "curve polynomial must use exactly two ordered variables.",
                {
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                        (-1, (0, 0)),
                    ),
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
                "Compute the projective closure of x^2 + y^2 - 1; the affine "
                "axis must have two variables and leave z available.",
                {
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (0, 2)),
                        (-1, (0, 0)),
                    ),
                },
            ),
        ),
    ),
    _op(
        "algebraic_geometry.conic.rational_parametrization.compute",
        "Parametrize a smooth rational conic",
        "Construct canonical rational coordinate functions, their inverse chart, "
        "the finite affine-denominator locus, and the exceptional source point "
        "from a smooth rational affine conic with a supplied rational point.",
        RationalConicParametrizationRequest,
        RationalConicParametrizationResult,
        compute_rational_conic_parametrization,
        "algebraic-geometry",
        "conic",
        "rational-parametrization",
        "exact",
        examples=(
            example(
                "smooth_conic_from_point",
                "Parametrize x^2 + x*y - y^2 - 1 = 0 from (1,0); the "
                "polynomial must have smooth projective closure and the checked "
                "point must lie on it.",
                {
                    "polynomial": _polynomial(
                        ("x", "y"),
                        (1, (2, 0)),
                        (1, (1, 1)),
                        (-1, (0, 2)),
                        (-1, (0, 0)),
                    ),
                    "point": {
                        "variables": ["x", "y"],
                        "values": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    "parameter": "t",
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
                "Extract the z=1 chart of x^2 + y^2 - z^2; the canonical "
                "projective polynomial must be homogeneous in three variables.",
                {
                    "polynomial": _polynomial(
                        ("x", "y", "z"),
                        (1, (2, 0, 0)),
                        (1, (0, 2, 0)),
                        (-1, (0, 0, 2)),
                    ),
                    "chart_variable": "z",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
