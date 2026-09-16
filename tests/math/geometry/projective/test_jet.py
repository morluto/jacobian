"""Tests for exact plane-curve first jets at rational projective points."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.projective._jet import (
    FIRST_JET_OPERATION,
    PlaneCurveJetRequest,
    plane_curve_first_jet,
)
from jacobian.math.geometry.projective._tools import TOOLS
from jacobian.math.geometry.projective.coordinates._models import (
    RationalProjectivePoint,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _term(coefficient: int, exponents: tuple[int, int, int]) -> RationalPolynomialTerm:
    return RationalPolynomialTerm(
        coefficient=_rational(coefficient), exponents=exponents
    )


def _curve(*terms: RationalPolynomialTerm) -> RationalPolynomial:
    ordered = tuple(sorted(terms, key=lambda term: term.exponents, reverse=True))
    return RationalPolynomial(
        domain="QQ",
        variables=("x", "y", "z"),
        polynomial=SparseRationalPolynomial(terms=ordered),
    )


def _point(*coords: int) -> RationalProjectivePoint:
    return RationalProjectivePoint(coordinates=tuple(_rational(c) for c in coords))


def _cusp() -> RationalPolynomial:
    # y^2 z - x^3, in descending lexicographic term order.
    return _curve(_term(-1, (3, 0, 0)), _term(1, (0, 2, 1)))


class TestKnownAnswer:
    def test_cusp_origin_is_singular(self) -> None:
        result = plane_curve_first_jet(_cusp(), _point(0, 0, 1))
        assert result.value.as_fraction() == 0
        assert [p.as_fraction() for p in result.partials] == [Fraction(0)] * 3
        assert result.chart_index == 2
        assert result.status == "ON_CURVE_SINGULAR_OR_HIGHER"

    def test_smooth_point_is_simple(self) -> None:
        # [-1:1:1]: (-1)^3 + ... : F = y^2 z - x^3 -> 1 + 1 = ... check: 1*1 - (-1) = 2? no.
        # Use [1:1:1]: 1 - 1 = 0 on curve; gradient (-3, 2, 1) nonzero.
        result = plane_curve_first_jet(_cusp(), _point(1, 1, 1))
        assert result.value.as_fraction() == 0
        assert [p.as_fraction() for p in result.partials] == [
            Fraction(-3),
            Fraction(2),
            Fraction(1),
        ]
        assert result.status == "ON_CURVE_SIMPLE"

    def test_off_curve_point(self) -> None:
        result = plane_curve_first_jet(_cusp(), _point(1, 0, 1))
        assert result.value.as_fraction() == Fraction(-1)
        assert result.status == "OFF_CURVE"


class TestBoundary:
    def test_line_on_curve_simple(self) -> None:
        line = _curve(_term(1, (1, 0, 0)), _term(1, (0, 1, 0)), _term(1, (0, 0, 1)))
        result = plane_curve_first_jet(line, _point(1, -1, 0))
        assert result.status == "ON_CURVE_SIMPLE"
        assert result.chart_index == 0

    def test_conic_off_curve(self) -> None:
        conic = _curve(_term(1, (2, 0, 0)), _term(1, (0, 2, 0)), _term(-1, (0, 0, 2)))
        result = plane_curve_first_jet(conic, _point(0, 0, 1))
        assert result.value.as_fraction() == Fraction(-1)
        assert result.status == "OFF_CURVE"

    def test_first_nonzero_chart_selection(self) -> None:
        line = _curve(_term(1, (1, 0, 0)))
        result = plane_curve_first_jet(line, _point(0, 2, 3))
        assert result.chart_index == 1
        assert [c.as_fraction() for c in result.affine_point] == [
            Fraction(0),
            Fraction(3, 2),
        ]


class TestAdversarial:
    def test_nonhomogeneous_rejected(self) -> None:
        mixed = _curve(_term(1, (2, 0, 0)), _term(1, (0, 0, 1)))
        with pytest.raises(OperationDomainValidationError):
            plane_curve_first_jet(mixed, _point(1, 0, 1))

    def test_zero_point_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            plane_curve_first_jet(_cusp(), _point(0, 0, 0))

    def test_wrong_variable_count_rejected(self) -> None:
        poly = RationalPolynomial(
            domain="QQ",
            variables=("x", "y"),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(coefficient=_rational(1), exponents=(2, 0)),
                    RationalPolynomialTerm(coefficient=_rational(-1), exponents=(0, 2)),
                )
            ),
        )
        with pytest.raises(OperationDomainValidationError):
            plane_curve_first_jet(poly, _point(1, 1, 1))

    def test_zero_polynomial_rejected(self) -> None:
        poly = RationalPolynomial(
            domain="QQ",
            variables=("x", "y", "z"),
            polynomial=SparseRationalPolynomial(terms=()),
        )
        with pytest.raises(OperationDomainValidationError):
            plane_curve_first_jet(poly, _point(1, 0, 0))


class TestDefiningInvariant:
    def test_point_rescaling_invariance(self) -> None:
        first = plane_curve_first_jet(_cusp(), _point(1, 1, 1))
        scaled = RationalProjectivePoint(
            coordinates=(
                CanonicalRational(num=2, den=1),
                CanonicalRational(num=2, den=1),
                CanonicalRational(num=2, den=1),
            )
        )
        second = plane_curve_first_jet(_cusp(), scaled)
        assert second.status == first.status == "ON_CURVE_SIMPLE"
        assert [p.as_fraction() for p in second.partials] == [
            p.as_fraction() * 4 for p in first.partials
        ]

    def test_polynomial_rescaling_invariance(self) -> None:
        doubled = _curve(_term(-2, (3, 0, 0)), _term(2, (0, 2, 1)))
        result = plane_curve_first_jet(doubled, _point(1, 1, 1))
        assert result.status == "ON_CURVE_SIMPLE"
        assert result.value.as_fraction() == 0

    def test_dehomogenized_replay(self) -> None:
        result = plane_curve_first_jet(_cusp(), _point(1, 1, 1))
        chart = result.chart_index
        coords = [Fraction(1), Fraction(1), Fraction(1)]
        scale = coords[chart]
        affine = [c / scale for j, c in enumerate(coords) if j != chart]
        assert [c.as_fraction() for c in result.affine_point] == affine


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        request = PlaneCurveJetRequest(polynomial=_cusp(), point=_point(0, 0, 1))
        assert FIRST_JET_OPERATION.run(request) == plane_curve_first_jet(
            _cusp(), _point(0, 0, 1)
        )

    def test_operation_is_published(self) -> None:
        assert "projective_geometry.plane_curve.first_jet.compute" in {
            tool.operation_id for tool in TOOLS
        }
