"""Tests for commutative algebra operations."""

from jacobian.math.commutative_algebra_ops._models import (
    IdealQuotientRequest,
    IdealRadicalRequest,
    IdealRequest,
)
from jacobian.math.commutative_algebra_ops._operations import (
    compute_ideal_quotient,
    compute_ideal_radical,
    compute_ideal_radical_membership,
)
from jacobian.math.commutative_algebra_ops._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial.ideal.radical.compute",
        "polynomial.ideal.radical_membership.decide",
        "polynomial.ideal.quotient.compute",
    }


def test_ideal_radical_basic() -> None:
    request = IdealRadicalRequest(
        variables=("x", "y"), generators=("x**2", "x*y")
    )
    result = compute_ideal_radical(request)
    assert len(result.generators) == 2
    assert "x**2" in result.generators


def test_ideal_radical_membership() -> None:
    request = IdealRequest(variables=("x",), generators=("x**2",))
    result = compute_ideal_radical_membership(request)
    assert result.in_radical is False


def test_ideal_quotient() -> None:
    request = IdealQuotientRequest(
        variables=("x", "y"),
        generators_a=("x**2", "x*y"),
        generators_b=("x",),
    )
    result = compute_ideal_quotient(request)
    assert isinstance(result.generators, tuple)
