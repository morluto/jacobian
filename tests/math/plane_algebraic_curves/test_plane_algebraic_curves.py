"""Tests for plane algebraic curve operations."""

import pytest
from pydantic import ValidationError

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
    request = AffineCurveRequest(variables=("x", "y"), polynomial="x**2 + y**2 - 1")
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


# --- Issue 6: round-trip, non-polynomial rejection, variable named "z",
#     constant polynomial -----------------------------------------------


def test_homogenize_dehomogenize_round_trip() -> None:
    """Homogenizing an affine curve then taking the z=1 chart recovers it."""
    affine = "x**3 - 2*x*y + y - 7"
    request = ProjectiveClosureRequest(variables=("x", "y"), polynomial=affine)
    closure = compute_projective_closure(request)
    assert closure.variables == ("x", "y", "z")
    chart = compute_affine_chart(
        AffineChartRequest(
            variables=closure.variables,
            polynomial=closure.polynomial,
            chart_variable="z",
        )
    )
    recovered = chart.polynomial.replace(" ", "")
    original = affine.replace(" ", "")
    assert recovered == original
    assert chart.variables == ("x", "y")


def test_non_polynomial_expression_is_rejected() -> None:
    """A non-polynomial expression must be rejected at parse time."""
    for model in (AffineCurveRequest, ProjectiveClosureRequest, AffineChartRequest):
        if model is AffineChartRequest:
            kwargs = {
                "variables": ("x", "y", "z"),
                "polynomial": "sin(x) + y",
                "chart_variable": "z",
            }
        else:
            kwargs = {"variables": ("x", "y"), "polynomial": "sin(x) + y"}
        with pytest.raises(ValidationError, match="polynomial"):
            model(**kwargs)


def test_unparseable_expression_is_rejected() -> None:
    with pytest.raises(ValidationError, match="polynomial"):
        AffineCurveRequest(variables=("x", "y"), polynomial="x +* y")


def test_undeclared_variable_is_rejected() -> None:
    with pytest.raises(ValidationError, match="undeclared"):
        AffineCurveRequest(variables=("x", "y"), polynomial="x + t")


def test_duplicate_variable_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        AffineCurveRequest(variables=("x", "x"), polynomial="x + 1")


def test_variable_named_z_rejected_in_projective_closure() -> None:
    """The homogenizing coordinate 'z' must not collide with a user variable."""
    with pytest.raises(ValidationError, match="homogenizing"):
        ProjectiveClosureRequest(variables=("x", "z"), polynomial="x**2 - z")


def test_constant_polynomial_is_not_a_valid_curve() -> None:
    request = AffineCurveRequest(variables=("x", "y"), polynomial="5")
    result = compute_affine_curve_check(request)
    assert result.is_valid is False
    assert result.degree == 0


def test_zero_polynomial_is_not_a_valid_curve() -> None:
    request = AffineCurveRequest(variables=("x", "y"), polynomial="0")
    result = compute_affine_curve_check(request)
    assert result.is_valid is False
    assert result.degree == 0


def test_chart_variable_must_be_a_projective_variable() -> None:
    with pytest.raises(ValidationError, match="chart_variable must be"):
        AffineChartRequest(
            variables=("x", "y", "z"),
            polynomial="x**2 + y**2 - z**2",
            chart_variable="w",
        )
