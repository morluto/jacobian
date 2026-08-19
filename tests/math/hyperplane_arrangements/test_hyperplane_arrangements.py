"""Tests for hyperplane arrangement operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.hyperplane_arrangements._models import (
    ChamberCountRequest,
    CharacteristicPolynomialRequest,
    HyperplaneArrangementRequest,
    RationalHyperplane,
)
from jacobian.math.hyperplane_arrangements._operations import (
    compute_arrangement,
    compute_chamber_count,
    compute_characteristic_polynomial,
)
from jacobian.math.hyperplane_arrangements._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "arrangement.construct",
        "arrangement.characteristic_polynomial.compute",
        "arrangement.chamber_count.compute",
    }


def test_arrangement_central() -> None:
    request = HyperplaneArrangementRequest(
        ambient_dimension=2,
        hyperplanes=(
            {"coefficients": ("1", "0"), "constant": "0"},
            {"coefficients": ("0", "1"), "constant": "0"},
        ),
    )
    result = compute_arrangement(request)
    assert result.is_central is True
    assert result.hyperplane_count == 2


def test_arrangement_noncentral() -> None:
    request = HyperplaneArrangementRequest(
        ambient_dimension=2,
        hyperplanes=(
            {"coefficients": ("1", "0"), "constant": "0"},
            {"coefficients": ("0", "1"), "constant": "1"},
        ),
    )
    result = compute_arrangement(request)
    assert result.is_central is False


# --- Issue 1: characteristic polynomial must be monic and correct ---


def test_characteristic_polynomial_generic() -> None:
    request = CharacteristicPolynomialRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_characteristic_polynomial(request)
    assert result.degree == 2
    assert len(result.coefficients) == 3


def test_characteristic_polynomial_is_monic() -> None:
    """chi(t) must always be monic of degree n."""
    for n, m in [(1, 1), (2, 2), (2, 3), (3, 4), (3, 2), (4, 6)]:
        request = CharacteristicPolynomialRequest(
            ambient_dimension=n, hyperplane_count=m
        )
        result = compute_characteristic_polynomial(request)
        assert result.degree == n
        assert result.coefficients[-1] == "1", (
            f"leading coefficient must be 1 for n={n}, m={m}"
        )


def test_characteristic_polynomial_n2_m2() -> None:
    """n=2, m=2: chi(t) = t^2 - 2t + 1 (not 4t^2 - 2t + 1)."""
    request = CharacteristicPolynomialRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_characteristic_polynomial(request)
    assert result.coefficients == ("1", "-2", "1")


# --- Issue 2: chamber count must use central formula ---


def test_chamber_count_generic() -> None:
    request = ChamberCountRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_chamber_count(request)
    assert result.chamber_count == 4


def test_chamber_count_central_m_gt_n() -> None:
    """n=2, m=3: 6 regions (not 7)."""
    request = ChamberCountRequest(ambient_dimension=2, hyperplane_count=3)
    result = compute_chamber_count(request)
    assert result.chamber_count == 6


def test_chamber_count_zaslavsky_consistency() -> None:
    """regions = (-1)^n * chi(-1) must hold for several (n, m) pairs."""
    for n, m in [(1, 1), (2, 2), (2, 3), (3, 4), (3, 5), (4, 6)]:
        chi_result = compute_characteristic_polynomial(
            CharacteristicPolynomialRequest(ambient_dimension=n, hyperplane_count=m)
        )
        count_result = compute_chamber_count(
            ChamberCountRequest(ambient_dimension=n, hyperplane_count=m)
        )
        coeffs = [int(c) for c in chi_result.coefficients]
        chi_neg1 = sum(v * (-1) ** i for i, v in enumerate(coeffs))
        zaslavsky = (-1) ** n * chi_neg1
        assert zaslavsky == count_result.chamber_count, (
            f"Zaslavsky mismatch for n={n}, m={m}: "
            f"{zaslavsky} != {count_result.chamber_count}"
        )


# --- Issue 3: validate hyperplane inputs ---


def test_rational_hyperplane_valid() -> None:
    hp = RationalHyperplane(coefficients=("1", "0"), constant="0")
    assert hp.coefficients == ("1", "0")
    assert hp.constant == "0"


def test_rational_hyperplane_rejects_non_rational() -> None:
    with pytest.raises(ValidationError):
        RationalHyperplane(coefficients=("sqrt(2)", "0"), constant="0")


def test_rational_hyperplane_rejects_all_zero_coefficients() -> None:
    with pytest.raises(ValidationError):
        RationalHyperplane(coefficients=("0", "0"), constant="0")


def test_rational_hyperplane_rejects_non_rational_constant() -> None:
    with pytest.raises(ValidationError):
        RationalHyperplane(coefficients=("1", "0"), constant="abc")


def test_rational_hyperplane_accepts_negative_rationals() -> None:
    hp = RationalHyperplane(coefficients=("-1/2", "3/4"), constant="5/6")
    assert hp.coefficients == ("-1/2", "3/4")
    assert hp.constant == "5/6"
