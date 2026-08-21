"""Tests for plane algebraic curve operations."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
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
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...],
    *terms: tuple[int, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(coefficient, 1),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "algebraic_geometry.affine_plane_curve.check",
        "algebraic_geometry.plane_curve.projective_closure.compute",
        "algebraic_geometry.projective_curve.affine_chart.compute",
    }


def test_affine_curve_check_circle() -> None:
    request = AffineCurveRequest(
        polynomial=_polynomial(("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0)))
    )
    result = compute_affine_curve_check(request)
    assert result.is_valid is True
    assert result.degree == 2


def test_projective_closure_circle_is_canonical_polynomial() -> None:
    source = _polynomial(("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0)))
    result = compute_projective_closure(ProjectiveClosureRequest(polynomial=source))
    assert result.polynomial == _polynomial(
        ("x", "y", "z"),
        (1, (2, 0, 0)),
        (1, (0, 2, 0)),
        (-1, (0, 0, 2)),
    )


def test_affine_chart_circle_is_directly_composable() -> None:
    projective = _polynomial(
        ("x", "y", "z"),
        (1, (2, 0, 0)),
        (1, (0, 2, 0)),
        (-1, (0, 0, 2)),
    )
    result = compute_affine_chart(
        AffineChartRequest(polynomial=projective, chart_variable="z")
    )
    assert result.polynomial == _polynomial(
        ("x", "y"), (1, (2, 0)), (1, (0, 2)), (-1, (0, 0))
    )
    AffineCurveRequest(polynomial=result.polynomial)


def test_homogenize_dehomogenize_round_trip() -> None:
    affine = _polynomial(
        ("x", "y"),
        (1, (3, 0)),
        (-2, (1, 1)),
        (1, (0, 1)),
        (-7, (0, 0)),
    )
    closure = compute_projective_closure(ProjectiveClosureRequest(polynomial=affine))
    chart = compute_affine_chart(
        AffineChartRequest(polynomial=closure.polynomial, chart_variable="z")
    )
    assert chart.polynomial == affine


def test_expression_strings_are_not_a_public_polynomial_contract() -> None:
    for payload in (
        "sin(x) + y",
        "x +* y",
        "x + t",
        "__import__('os').getcwd()",
    ):
        with pytest.raises(ValidationError, match="polynomial"):
            AffineCurveRequest.model_validate({"polynomial": payload})


def test_duplicate_and_invalid_variable_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _polynomial(("x", "x"), (1, (1, 0)))
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        _polynomial(("", "y"), (1, (1, 0)))


def test_variable_named_z_rejected_in_projective_closure() -> None:
    with pytest.raises(ValidationError, match="homogenizing"):
        ProjectiveClosureRequest(
            polynomial=_polynomial(("x", "z"), (1, (2, 0)), (-1, (0, 1)))
        )


@pytest.mark.parametrize("constant", [0, 5])
def test_constant_polynomial_is_not_a_valid_curve(constant: int) -> None:
    terms = () if constant == 0 else ((constant, (0, 0)),)
    result = compute_affine_curve_check(
        AffineCurveRequest(polynomial=_polynomial(("x", "y"), *terms))
    )
    assert result.is_valid is False
    assert result.degree == 0


def test_chart_requires_three_variables_and_a_homogeneous_polynomial() -> None:
    with pytest.raises(ValidationError, match="exactly three"):
        AffineChartRequest(
            polynomial=_polynomial(("x", "y"), (1, (2, 0))),
            chart_variable="x",
        )
    with pytest.raises(ValidationError, match="homogeneous"):
        AffineChartRequest(
            polynomial=_polynomial(("x", "y", "z"), (1, (2, 0, 0)), (1, (0, 1, 0))),
            chart_variable="z",
        )


def test_chart_variable_must_be_on_the_projective_axis() -> None:
    with pytest.raises(ValidationError, match="must belong"):
        AffineChartRequest(
            polynomial=_polynomial(
                ("x", "y", "z"),
                (1, (2, 0, 0)),
                (1, (0, 2, 0)),
                (-1, (0, 0, 2)),
            ),
            chart_variable="w",
        )
