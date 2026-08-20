"""Tests for commutative algebra operations."""

from jacobian.math.commutative_algebra_ops._models import (
    IdealQuotientRequest,
    IdealRadicalMembershipRequest,
    IdealRadicalRequest,
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


# ---------------------------------------------------------------------------
# Radical: known-answer tests
# ---------------------------------------------------------------------------


def test_ideal_radical_xy() -> None:
    """Radical of <x^2, xy> is <x>."""
    request = IdealRadicalRequest(variables=("x", "y"), generators=("x**2", "x*y"))
    result = compute_ideal_radical(request)
    assert result.generators == ("x", "x")


def test_ideal_radical_already_radical() -> None:
    """Radical of <x^2 - 2> is <x^2 - 2> (already radical)."""
    request = IdealRadicalRequest(variables=("x",), generators=("x**2 - 2",))
    result = compute_ideal_radical(request)
    assert result.generators == ("x**2 - 2",)


def test_ideal_radical_square_free() -> None:
    """Radical of <x^2, y^2> is <x, y>."""
    request = IdealRadicalRequest(variables=("x", "y"), generators=("x**2", "y**2"))
    result = compute_ideal_radical(request)
    assert "x" in result.generators[0]
    assert "y" in result.generators[1]


# ---------------------------------------------------------------------------
# Radical membership: known-answer tests
# ---------------------------------------------------------------------------


def test_ideal_radical_membership_true() -> None:
    """x is in the radical of <x^2>."""
    request = IdealRadicalMembershipRequest(
        variables=("x",), generators=("x**2",), polynomial="x"
    )
    result = compute_ideal_radical_membership(request)
    assert result.in_radical is True


def test_ideal_radical_membership_false() -> None:
    """x+1 is NOT in the radical of <x^2>."""
    request = IdealRadicalMembershipRequest(
        variables=("x",), generators=("x**2",), polynomial="x + 1"
    )
    result = compute_ideal_radical_membership(request)
    assert result.in_radical is False


def test_ideal_radical_membership_multivariate() -> None:
    """x is in the radical of <x^2, x*y>, but y is not."""
    request_x = IdealRadicalMembershipRequest(
        variables=("x", "y"),
        generators=("x**2", "x*y"),
        polynomial="x",
    )
    assert compute_ideal_radical_membership(request_x).in_radical is True

    request_y = IdealRadicalMembershipRequest(
        variables=("x", "y"),
        generators=("x**2", "x*y"),
        polynomial="y",
    )
    assert compute_ideal_radical_membership(request_y).in_radical is False


# ---------------------------------------------------------------------------
# Ideal quotient: known-answer tests
# ---------------------------------------------------------------------------


def test_ideal_quotient_xy() -> None:
    """(<x^2, xy> : <x>) = <x, y>."""
    request = IdealQuotientRequest(
        variables=("x", "y"),
        generators_a=("x**2", "x*y"),
        generators_b=("x",),
    )
    result = compute_ideal_quotient(request)
    assert set(result.generators) == {"x", "y"}


def test_ideal_quotient_y() -> None:
    """(<x^2, xy> : <y>) = <x>."""
    request = IdealQuotientRequest(
        variables=("x", "y"),
        generators_a=("x**2", "x*y"),
        generators_b=("y",),
    )
    result = compute_ideal_quotient(request)
    assert result.generators == ("x",)


def test_ideal_quotient_by_whole_ring() -> None:
    """(I : <1>) = I."""
    request = IdealQuotientRequest(
        variables=("x",),
        generators_a=("x**2",),
        generators_b=("1",),
    )
    result = compute_ideal_quotient(request)
    assert result.generators == ("x**2",)


# ---------------------------------------------------------------------------
# Single-variable edge case
# ---------------------------------------------------------------------------


def test_ideal_radical_single_var() -> None:
    """Radical of <x^2> in Q[x] is <x>."""
    request = IdealRadicalRequest(variables=("x",), generators=("x**2",))
    result = compute_ideal_radical(request)
    assert result.generators == ("x",)
