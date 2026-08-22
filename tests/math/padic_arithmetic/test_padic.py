"""Tests for p-adic number theory operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.padic_arithmetic._models import (
    HenselRootRequest,
    IntegerPolynomial,
    PAdicRootsRequest,
)
from jacobian.math.padic_arithmetic._operations import (
    find_padic_roots,
    hensel_lift_root,
)


class TestHenselRootLifting:
    """Test Hensel root lifting."""

    def test_lift_simple_root(self):
        """Lift root 2 of x^2+1 mod 5 to mod 5^3."""
        poly = IntegerPolynomial(coefficients=(1, 0, 1))
        result = hensel_lift_root(
            HenselRootRequest(
                polynomial=poly, prime=5, root_mod_p=2, precision=3
            )
        )
        assert result.is_simple_root
        assert (result.lifted_root**2 + 1) % 125 == 0

    def test_lift_mod_p_squared(self):
        """Lift root to mod p^2."""
        poly = IntegerPolynomial(coefficients=(-1, 0, 0, 1))  # x^3 - 1
        result = hensel_lift_root(
            HenselRootRequest(
                polynomial=poly, prime=5, root_mod_p=1, precision=2
            )
        )
        assert result.is_simple_root
        assert (result.lifted_root**3 - 1) % 25 == 0

    def test_non_root_rejected(self):
        """A non-root mod p should be rejected."""
        poly = IntegerPolynomial(coefficients=(1, 0, 1))  # x^2 + 1
        with pytest.raises(ValidationError, match=r"f\(root_mod_p\) = 0"):
            hensel_lift_root(
                HenselRootRequest(
                    polynomial=poly, prime=5, root_mod_p=1, precision=2
                )
            )

    def test_root_in_range(self):
        """Lifted root should be in [0, p^k - 1]."""
        poly = IntegerPolynomial(coefficients=(3, 0, 1))  # x^2 + 3
        result = hensel_lift_root(
            HenselRootRequest(
                polynomial=poly, prime=7, root_mod_p=2, precision=4
            )
        )
        assert 0 <= result.lifted_root < 7**4
        assert (result.lifted_root**2 + 3) % (7**4) == 0


class TestPAdicRoots:
    """Test p-adic root finding."""

    def test_find_roots_x3_minus_1(self):
        """Find roots of x^3 - 1 mod 5^2."""
        poly = IntegerPolynomial(coefficients=(-1, 0, 0, 1))
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=5, precision=2)
        )
        assert result.root_count >= 1
        for root in result.roots:
            assert (root.root**3 - 1) % 25 == 0

    def test_find_roots_no_roots(self):
        """Find roots of x^2 + 1 mod 3 (no roots mod 3)."""
        poly = IntegerPolynomial(coefficients=(1, 0, 1))
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=3, precision=2)
        )
        assert result.root_count == 0

    def test_find_roots_x_squared_mod_5(self):
        """x^2 mod 5 has one multiple residue (0) and no simple roots."""
        poly = IntegerPolynomial(coefficients=(0, 0, 1))
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=5, precision=2)
        )
        assert result.root_count == 0
        assert result.multiple_residues == (0,)

    def test_composite_prime_rejected(self):
        """Composite moduli are rejected at the typed boundary."""
        poly = IntegerPolynomial(coefficients=(-1, 0, 1))
        with pytest.raises(ValidationError, match="prime modulus"):
            HenselRootRequest(
                polynomial=poly, prime=4, root_mod_p=1, precision=2
            )

    def test_multiple_root_lift_rejected(self):
        """f=x^2+5, p=5: r=0 is a multiple root; lifting is refused."""
        poly = IntegerPolynomial(coefficients=(5, 0, 1))
        with pytest.raises(ValidationError, match="simple root"):
            HenselRootRequest(
                polynomial=poly, prime=5, root_mod_p=0, precision=2
            )

    def test_all_roots_are_valid(self):
        """All returned roots should satisfy f(root) ≡ 0 (mod p^k)."""
        poly = IntegerPolynomial(coefficients=(-1, 0, 0, 1))  # x^3 - 1
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=7, precision=3)
        )
        for root in result.roots:
            assert (root.root**3 - 1) % (7**3) == 0
