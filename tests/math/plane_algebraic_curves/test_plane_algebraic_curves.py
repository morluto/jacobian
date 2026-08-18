"""Tests for plane algebraic curve operations."""

from jacobian.math.plane_algebraic_curves._models import (
    AffineChartRequest,
    AffineCurveRequest,
    ProjectiveClosureRequest,
)
from jacobian.math.plane_algebraic_curves._operations import (
    compute_affine_chart,
    compute_affine_curve_check,
    compute_projective_closure,
)
from jacobian.math.plane_algebraic_curves._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "algebraic_geometry.affine_plane_curve.check",
        "algebraic_geometry.plane_curve.projective_closure.compute",
        "algebraic_geometry.projective_curve.affine_chart.compute",
    }


def test_affine_curve_check_circle() -> None:
    request = AffineCurveRequest(
        variables=("x", "y"), polynomial="x**2 + y**2 - 1"
    )
    result = compute_affine_curve_check(request)
    assert result.is_valid is True
    assert result.degree == 2


def test_projective_closure_circle() -> None:
    request = ProjectiveClosureRequest(
        variables=("x", "y"), polynomial="x**2 + y**2 - 1"
    )
    result = compute_projective_closure(request)
    assert "z" in result.polynomial


def test_affine_chart_circle() -> None:
    request = AffineChartRequest(
        variables=("x", "y", "z"),
        polynomial="x**2 + y**2 - z**2",
        chart_variable="z",
    )
    result = compute_affine_chart(request)
    assert result.polynomial == "x**2 + y**2 - 1"
    assert result.variables == ("x", "y")
