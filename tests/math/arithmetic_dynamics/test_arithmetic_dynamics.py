"""Tests for arithmetic dynamics operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.arithmetic_dynamics._models import (
    CycleMultiplierRequest,
    DynatomicPolynomialRequest,
    FiniteFieldMapRequest,
    FixedPointEquationRequest,
    MapIterateRequest,
    OrbitPrefixRequest,
)
from jacobian.math.arithmetic_dynamics._operations import (
    compute_cycle_multiplier,
    compute_dynatomic_polynomial,
    compute_finite_field_map,
    compute_fixed_point_equation,
    compute_map_iterate,
    compute_orbit_prefix,
)


class TestMapIterate:
    def test_identity(self):
        """f^0 should be the identity (x)."""
        req = MapIterateRequest(coefficients=("1", "0", "1"), n=0)
        result = compute_map_iterate(req)
        assert result.n == 0
        # Identity is just x, so coefficients are [0, 1]
        assert result.coefficients[1] == "1"

    def test_first_iterate(self):
        """f^1 should equal f."""
        req = MapIterateRequest(coefficients=("1", "0", "1"), n=1)
        result = compute_map_iterate(req)
        # f(x) = x^2 + 1, coefficients low-to-high: [1, 0, 1]
        assert result.coefficients == ("1", "0", "1")

    def test_second_iterate(self):
        """f^2 for f(x) = x^2 + 1 should be (x^2+1)^2 + 1 = x^4 + 2x^2 + 2."""
        req = MapIterateRequest(coefficients=("1", "0", "1"), n=2)
        result = compute_map_iterate(req)
        # f(f(x)) = (x^2+1)^2 + 1 = x^4 + 2x^2 + 2
        # coefficients low-to-high: [2, 0, 2, 0, 1]
        assert result.degree == 4


class TestOrbitPrefix:
    def test_fixed_point(self):
        """Orbit of 0 under x^2 should be 0, 0, 0, ... (fixed point)."""
        req = OrbitPrefixRequest(coefficients=("0", "0", "1"), start="0", length=5)
        result = compute_orbit_prefix(req)
        assert result.orbit[0] == "0"
        assert result.first_repeat_index is not None

    def test_simple_orbit(self):
        """Orbit of 2 under x^2 should be 2, 4, 16, 256, ..."""
        req = OrbitPrefixRequest(coefficients=("0", "0", "1"), start="2", length=3)
        result = compute_orbit_prefix(req)
        assert result.orbit[0] == "2"
        assert result.orbit[1] == "4"
        assert result.orbit[2] == "16"


class TestFixedPointEquation:
    def test_simple(self):
        """f(x) = x^2, f(x) - x = x^2 - x."""
        req = FixedPointEquationRequest(coefficients=("0", "0", "1"), n=1)
        result = compute_fixed_point_equation(req)
        # x^2 - x, coefficients low-to-high: [0, -1, 1]
        assert result.coefficients[1] == "-1"
        assert result.coefficients[2] == "1"


class TestDynatomicPolynomial:
    def test_n1(self):
        """For n=1, Phi*_1 = f(x) - x."""
        req = DynatomicPolynomialRequest(coefficients=("0", "0", "1"), n=1)
        result = compute_dynatomic_polynomial(req)
        # Phi*_1 = x^2 - x, coefficients: [0, -1, 1]
        assert result.n == 1
        assert result.coefficients[2] == "1"


class TestCycleMultiplier:
    def test_fixed_point(self):
        """For f(x) = x^2, the fixed point 0 has multiplier f'(0) = 0."""
        req = CycleMultiplierRequest(coefficients=("0", "0", "1"), cycle=("0",))
        result = compute_cycle_multiplier(req)
        assert result.multiplier == "0"

    def test_fixed_point_1(self):
        """For f(x) = x^2, the fixed point 1 has multiplier f'(1) = 2."""
        req = CycleMultiplierRequest(coefficients=("0", "0", "1"), cycle=("1",))
        result = compute_cycle_multiplier(req)
        assert result.multiplier == "2"


class TestFiniteFieldMap:
    def test_x2_mod_5(self):
        """Functional graph of x^2 over GF(5)."""
        req = FiniteFieldMapRequest(prime=5, coefficients=("0", "0", "1"))
        result = compute_finite_field_map(req)
        assert result.prime == 5
        assert len(result.edges) == 5
        # 0 -> 0, 1 -> 1, 2 -> 4, 3 -> 4, 4 -> 1
        edge_map = dict(result.edges)
        assert edge_map[0] == 0
        assert edge_map[1] == 1
        assert edge_map[2] == 4

    def test_rejects_non_prime(self):
        with pytest.raises(ValidationError, match="prime"):
            FiniteFieldMapRequest(prime=4, coefficients=("1",))


class TestValidation:
    def test_negative_n_rejected(self):
        with pytest.raises(ValidationError):
            MapIterateRequest(coefficients=("1", "0", "1"), n=-1)

    def test_negative_orbit_length(self):
        with pytest.raises(ValidationError):
            OrbitPrefixRequest(coefficients=("1",), start="0", length=-1)
