"""Tests for hyperplane arrangement operations."""

from jacobian.math.hyperplane_arrangements._models import (
    ChamberCountRequest,
    CharacteristicPolynomialRequest,
    HyperplaneArrangementRequest,
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


def test_characteristic_polynomial_generic() -> None:
    request = CharacteristicPolynomialRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_characteristic_polynomial(request)
    assert result.degree == 2
    assert len(result.coefficients) == 3


def test_chamber_count_generic() -> None:
    request = ChamberCountRequest(ambient_dimension=2, hyperplane_count=2)
    result = compute_chamber_count(request)
    assert result.chamber_count == 4
