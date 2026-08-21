"""Tests for approximation theory operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.approximation_theory._models import (
    LagrangeBasisRequest,
    LagrangeInterpolationRequest,
    RationalNodeSet,
)
from jacobian.math.approximation_theory._operations import (
    compute_lagrange_basis,
    compute_lagrange_interpolation,
)


def _node(num: str, den: str = "1") -> dict:
    return {"num": num, "den": den}


class TestDerivedGrowthBudget:
    """Admission must keep derived results inside the canonical digit limit."""

    def test_huge_denominator_nodes_rejected(self):
        # (10^2000 + k)/10^2000 is already reduced and has a 2001-digit
        # denominator per node; four such nodes blow the component budget.
        nodes = [
            _node(str(10**2000 + k), "1" + "0" * 2000) for k in (1, 3, 7, 9)
        ]
        with pytest.raises(ValidationError, match="component budget"):
            RationalNodeSet(nodes=tuple(CanonicalRational.model_validate(n) for n in nodes))

    def test_many_moderate_nodes_accepted(self):
        primes = [
            11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
            149, 151,
        ]
        nodes = [_node(str(p), str(10**14)) for p in primes]
        request = LagrangeBasisRequest(
            nodes=RationalNodeSet(
                nodes=tuple(CanonicalRational.model_validate(n) for n in nodes)
            )
        )
        result = compute_lagrange_basis(request)
        assert result.node_count == 32


class TestLagrangeBasis:
    """Test Lagrange basis computation."""

    def test_two_nodes(self):
        """Basis for {0, 1} should give l_0 = 1 - x, l_1 = x."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1")))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        assert result.node_count == 2
        assert result.basis[0].index == 0
        assert result.basis[1].index == 1

    def test_three_nodes(self):
        """Basis for {0, 1, 2} has three polynomials of degree 2."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1"), _node("2")))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        assert result.node_count == 3
        for bp in result.basis:
            assert len(bp.coefficients) == 3  # degree 2

    def test_barycentric_weights(self):
        """Barycentric weights for {0, 1, 2} are 1/2, -1, 1/2."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1"), _node("2")))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        weights = [bp.barycentric_weight.as_fraction() for bp in result.basis]
        assert weights == [Fraction(1, 2), Fraction(-1), Fraction(1, 2)]

    def test_basis_partition_of_unity(self):
        """Sum of basis polynomials equals 1."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1"), _node("2")))
        result = compute_lagrange_basis(LagrangeBasisRequest(nodes=nodes))
        max_len = max(len(bp.coefficients) for bp in result.basis)
        total = [Fraction(0)] * max_len
        for bp in result.basis:
            for i, c in enumerate(bp.coefficients):
                total[i] += c.as_fraction()
        assert total[0] == 1
        assert all(c == 0 for c in total[1:])


class TestLagrangeInterpolation:
    """Test Lagrange interpolation."""

    def test_two_points(self):
        """Interpolate through (0, 1), (1, 2) → x + 1."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1")))
        values = (_node("1"), _node("2"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(1), Fraction(1)]

    def test_three_points(self):
        """Interpolate through (0, 1), (1, 3), (2, 9) → 2x^2 + 1."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1"), _node("2")))
        values = (_node("1"), _node("3"), _node("9"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(1), Fraction(0), Fraction(2)]

    def test_rational_nodes(self):
        """Interpolate through (0, 0), (1/2, 1/4), (1, 1) → x^2."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1", "2"), _node("1")))
        values = (_node("0"), _node("1", "4"), _node("1"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(0), Fraction(0), Fraction(1)]

    def test_constant_interpolation(self):
        """Interpolate constant values → constant polynomial."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1"), _node("2")))
        values = (_node("5"), _node("5"), _node("5"))
        result = compute_lagrange_interpolation(
            LagrangeInterpolationRequest(nodes=nodes, values=values)
        )
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(5)]

    def test_mismatched_lengths_rejected(self):
        """Values length must match nodes length."""
        nodes = RationalNodeSet(nodes=(_node("0"), _node("1")))
        with pytest.raises(ValueError, match="same length"):
            LagrangeInterpolationRequest(
                nodes=nodes,
                values=(_node("1"), _node("2"), _node("3")),
            )
