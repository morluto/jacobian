"""Plane algebraic curve operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.algebraic_curves._models import (
    AffineChartRequest,
    AffineChartResult,
    AffineCurveRequest,
    AffineCurveResult,
    ProjectiveClosureRequest,
    ProjectiveClosureResult,
    ProjectivePlaneCurveSingularityProfile,
    ProjectivePlaneCurveSingularityRequest,
    RationalConicParametrizationRequest,
    RationalConicParametrizationResult,
)
from jacobian.math.geometry.algebraic_curves.operations import (
    affine_chart,
    affine_curve_check,
    projective_closure,
    rational_conic_parametrization,
    singularity_profile,
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


def compute_projective_plane_curve_singularity_profile(
    request: ProjectivePlaneCurveSingularityRequest,
) -> ProjectivePlaneCurveSingularityProfile:
    """Compute one exact source-bound global projective singular locus."""

    return singularity_profile(request.polynomial, request.resource_budget)


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="algebraic_geometry.projective_plane_curve.singularity_profile.compute",
        title="Compute a projective plane-curve singularity profile",
        description="Compute the complete saturated projective Jacobian locus over the "
        "algebraic closure of QQ for one bounded homogeneous ternary polynomial. "
        "The result distinguishes the unit-ideal smooth case, a complete finite "
        "family of exact embedded number-field points, a positive-dimensional "
        "locus, and operational noncompletion.",
        request_type=ProjectivePlaneCurveSingularityRequest,
        result_type=ProjectivePlaneCurveSingularityProfile,
        run=compute_projective_plane_curve_singularity_profile,
        tags=("algebraic-geometry", "projective-curve", "singular-locus", "exact"),
        examples=(
            OperationExample(
                name="conjugate_singularities",
                description="Find both non-rational singular points [1:i:0] and [1:-i:0] "
                "of Z*(X^2+Y^2+Z^2), retaining their distinct exact embeddings.",
                input={
                    "polynomial": _polynomial(
                        ("X", "Y", "Z"),
                        (1, (2, 0, 1)),
                        (1, (0, 2, 1)),
                        (1, (0, 0, 3)),
                    )
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_geometry.affine_plane_curve.check",
        title="Check an affine plane curve",
        description="Check that a polynomial defines a valid affine plane curve f(x,y)=0 "
        "and return its degree.",
        request_type=AffineCurveRequest,
        result_type=AffineCurveResult,
        run=compute_affine_curve_check,
        tags=("algebraic-geometry", "affine-curve", "exact"),
        examples=(
            OperationExample(
                name="circle",
                description="Check the unit circle x^2 + y^2 - 1 = 0; an affine plane "
                "curve polynomial must use exactly two ordered variables.",
                input={
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
    MathTool(
        operation_id="algebraic_geometry.plane_curve.projective_closure.compute",
        title="Compute the projective closure of an affine curve",
        description="Homogenize an affine plane curve to obtain its projective closure.",
        request_type=ProjectiveClosureRequest,
        result_type=ProjectiveClosureResult,
        run=compute_projective_closure,
        tags=("algebraic-geometry", "projective-closure", "exact"),
        examples=(
            OperationExample(
                name="circle_closure",
                description="Compute the projective closure of x^2 + y^2 - 1; the affine "
                "axis must have two variables and leave z available.",
                input={
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
    MathTool(
        operation_id="algebraic_geometry.conic.rational_parametrization.compute",
        title="Parametrize a smooth rational conic",
        description="Construct canonical rational coordinate functions, their inverse chart, "
        "the finite affine-denominator locus, and the exceptional source point "
        "from a smooth rational affine conic with a supplied rational point.",
        request_type=RationalConicParametrizationRequest,
        result_type=RationalConicParametrizationResult,
        run=compute_rational_conic_parametrization,
        tags=("algebraic-geometry", "conic", "rational-parametrization", "exact"),
        examples=(
            OperationExample(
                name="smooth_conic_from_point",
                description="Parametrize x^2 + x*y - y^2 - 1 = 0 from (1,0); the "
                "polynomial must have smooth projective closure and the checked "
                "point must lie on it.",
                input={
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
    MathTool(
        operation_id="algebraic_geometry.projective_curve.affine_chart.compute",
        title="Extract an affine chart from a projective curve",
        description="Dehomogenize a projective curve at the given chart variable by "
        "setting that variable to 1.",
        request_type=AffineChartRequest,
        result_type=AffineChartResult,
        run=compute_affine_chart,
        tags=("algebraic-geometry", "affine-chart", "exact"),
        examples=(
            OperationExample(
                name="chart_z",
                description="Extract the z=1 chart of x^2 + y^2 - z^2; the canonical "
                "projective polynomial must be homogeneous in three variables.",
                input={
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
