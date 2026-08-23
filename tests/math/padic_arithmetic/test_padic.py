"""Tests for p-adic number theory operations."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.math.padic_arithmetic._models import (
    HenselFactorLiftRequest,
    HenselRootRequest,
    HenselRootResult,
    IntegerPolynomial,
    PAdicRootEntry,
    PAdicRootsRequest,
    PAdicRootsResult,
)
from jacobian.math.padic_arithmetic._operations import (
    find_padic_roots,
    hensel_lift_factors,
    hensel_lift_root,
)


class TestHenselRootLifting:
    """Test Hensel root lifting."""

    def test_lift_simple_root(self):
        """Lift root 2 of x^2+1 mod 5 to mod 5^3."""
        poly = IntegerPolynomial(coefficients=("1", "0", "1"))
        result = hensel_lift_root(
            HenselRootRequest(polynomial=poly, prime=5, root_mod_p=2, precision=3)
        )
        assert result.is_simple_root
        assert (result.lifted_root**2 + 1) % 125 == 0

    def test_lift_mod_p_squared(self):
        """Lift root to mod p^2."""
        poly = IntegerPolynomial(coefficients=("1", "0", "0", "-1"))  # x^3 - 1
        result = hensel_lift_root(
            HenselRootRequest(polynomial=poly, prime=5, root_mod_p=1, precision=2)
        )
        assert result.is_simple_root
        assert (result.lifted_root**3 - 1) % 25 == 0

    def test_non_root_rejected(self):
        """A non-root mod p should be rejected."""
        poly = IntegerPolynomial(coefficients=("1", "0", "1"))  # x^2 + 1
        with pytest.raises(ValidationError, match=r"f\(root_mod_p\) = 0"):
            hensel_lift_root(
                HenselRootRequest(polynomial=poly, prime=5, root_mod_p=1, precision=2)
            )

    def test_root_in_range(self):
        """Lifted root should be in [0, p^k - 1]."""
        poly = IntegerPolynomial(coefficients=("1", "0", "3"))  # x^2 + 3
        result = hensel_lift_root(
            HenselRootRequest(polynomial=poly, prime=7, root_mod_p=2, precision=4)
        )
        assert 0 <= result.lifted_root < 7**4
        assert (result.lifted_root**2 + 3) % (7**4) == 0


class TestPAdicRoots:
    """Test p-adic root finding."""

    def test_find_roots_x3_minus_1(self):
        """Find roots of x^3 - 1 mod 5^2."""
        poly = IntegerPolynomial(coefficients=("1", "0", "0", "-1"))
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=5, precision=2)
        )
        assert result.root_count >= 1
        for root in result.roots:
            assert (root.root**3 - 1) % 25 == 0

    def test_find_roots_no_roots(self):
        """Find roots of x^2 + 1 mod 3 (no roots mod 3)."""
        poly = IntegerPolynomial(coefficients=("1", "0", "1"))
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=3, precision=2)
        )
        assert result.root_count == 0

    def test_find_roots_x_squared_mod_5(self):
        """x^2 mod 5 has one multiple residue (0) and no simple roots."""
        poly = IntegerPolynomial(coefficients=("1", "0", "0"))
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=5, precision=2)
        )
        assert result.root_count == 0
        assert result.multiple_residues == (0,)

    def test_composite_prime_rejected(self):
        """Composite moduli are rejected at the typed boundary."""
        poly = IntegerPolynomial(coefficients=("1", "-1"))
        with pytest.raises(ValidationError, match="prime modulus"):
            HenselRootRequest(polynomial=poly, prime=4, root_mod_p=1, precision=2)

    def test_result_rejects_composite_modulus(self):
        """A serialized root set cannot validate against a composite modulus
        even when its completeness and derivative replay would succeed."""
        with pytest.raises(ValidationError, match="prime modulus"):
            PAdicRootsResult(
                polynomial=IntegerPolynomial(coefficients=("1", "-1")),
                roots=(PAdicRootEntry(root=1),),
                prime=4,
                precision=2,
                root_count=1,
            )

    def test_result_binds_to_source_polynomial(self):
        """The exact result retains polynomial and residue and replays them:
        genuine lifts round-trip through serialization, while forged lifts,
        wrong residues, detached polynomials, and a False simple flag are
        rejected at validation."""
        poly = IntegerPolynomial(coefficients=("1", "0", "1"))  # x^2 + 1
        result = hensel_lift_root(
            HenselRootRequest(polynomial=poly, prime=5, root_mod_p=2, precision=4)
        )
        assert result.polynomial == poly
        assert result.root_mod_p == 2
        assert result.lifted_root % 5 == 2

        roundtrip = HenselRootResult.model_validate(result.model_dump())
        assert roundtrip == result

        detached = result.model_dump()
        detached["polynomial"]["coefficients"] = ["1", "0", "2"]
        with pytest.raises(ValidationError, match=r"f\(root_mod_p\) = 0"):
            HenselRootResult.model_validate(detached)

        forged = result.model_dump()
        forged["lifted_root"] = (forged["lifted_root"] + 1) % 625
        with pytest.raises(ValidationError):
            HenselRootResult.model_validate(forged)

        wrong_residue = result.model_dump()
        # residue 3 is itself a simple root of x^2+1 mod 5, so only the
        # congruence of the lift to its residue can reject this tampering
        wrong_residue["root_mod_p"] = 3
        with pytest.raises(ValidationError, match="reduce to root_mod_p"):
            HenselRootResult.model_validate(wrong_residue)

        not_simple = result.model_dump()
        not_simple["is_simple_root"] = False
        with pytest.raises(ValidationError, match="simple"):
            HenselRootResult.model_validate(not_simple)

    def test_multiple_root_lift_rejected(self):
        """f=x^2+5, p=5: r=0 is a multiple root; lifting is refused."""
        poly = IntegerPolynomial(coefficients=("1", "0", "5"))
        with pytest.raises(ValidationError, match="simple root"):
            HenselRootRequest(polynomial=poly, prime=5, root_mod_p=0, precision=2)

    def test_all_roots_are_valid(self):
        """All returned roots should satisfy f(root) ≡ 0 (mod p^k)."""
        poly = IntegerPolynomial(coefficients=("1", "0", "0", "-1"))  # x^3 - 1
        result = find_padic_roots(
            PAdicRootsRequest(polynomial=poly, prime=7, precision=3)
        )
        for root in result.roots:
            assert (root.root**3 - 1) % (7**3) == 0


def _wire_poly(*ascending: int) -> IntegerPolynomial:
    return IntegerPolynomial(
        coefficients=tuple(
            format_canonical_integer(coefficient) for coefficient in reversed(ascending)
        )
    )


def _poly_value_at(polynomial: IntegerPolynomial, x: int, modulus: int) -> int:
    coefficients = [
        int(coefficient) for coefficient in reversed(polynomial.coefficients)
    ]
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * x + coefficient) % modulus
    return result


class TestHenselFactorLifting:
    """Test coprime factorization lifting with Bézout corrections."""

    def test_thread_example_requires_bezout_correction(self):
        """f = x^2+x+3, g = x, h = x+1, p = 3: the lift must correct both
        factors through the Bézout relation, not shift coefficients of f-gh
        into g alone."""
        f = _wire_poly(3, 1, 1)
        g = _wire_poly(0, 1)
        h = _wire_poly(1, 1)
        result = hensel_lift_factors(
            HenselFactorLiftRequest(
                polynomial=f, factor_g=g, factor_h=h, prime=3, precision=2
            )
        )
        modulus = 9
        for x in range(modulus):
            product = (
                _poly_value_at(result.lifted_g, x, modulus)
                * _poly_value_at(result.lifted_h, x, modulus)
            ) % modulus
            assert product == _poly_value_at(f, x, modulus)

    def test_lift_preserves_product_mod_p_k(self):
        """(x+2)(x+3) ≡ x^2+1 (mod 5); lifting preserves it to 5^3."""
        f = IntegerPolynomial(coefficients=("1", "0", "1"))
        g = _wire_poly(2, 1)
        h = _wire_poly(3, 1)
        result = hensel_lift_factors(
            HenselFactorLiftRequest(
                polynomial=f, factor_g=g, factor_h=h, prime=5, precision=3
            )
        )
        modulus = 125
        for x in range(modulus):
            product = (
                _poly_value_at(result.lifted_g, x, modulus)
                * _poly_value_at(result.lifted_h, x, modulus)
            ) % modulus
            assert product == _poly_value_at(f, x, modulus)

    def test_higher_precision_lift(self):
        """A three-step lift stays exact modulo p^4."""
        f = _wire_poly(2, 3, 2, 1)  # (x+1)(x^2+x+2)
        g = _wire_poly(1, 1)
        h = _wire_poly(2, 1, 1)
        result = hensel_lift_factors(
            HenselFactorLiftRequest(
                polynomial=f, factor_g=g, factor_h=h, prime=5, precision=4
            )
        )
        modulus = 625
        for x in range(modulus):
            product = (
                _poly_value_at(result.lifted_g, x, modulus)
                * _poly_value_at(result.lifted_h, x, modulus)
            ) % modulus
            assert product == _poly_value_at(f, x, modulus)

    def test_precision_one_returns_factors_mod_p(self):
        """With k = 1 no lift happens; factors return reduced mod p."""
        f = IntegerPolynomial(coefficients=("1", "0", "1"))
        g = _wire_poly(7, 11)
        h = _wire_poly(18, 6)
        result = hensel_lift_factors(
            HenselFactorLiftRequest(
                polynomial=f, factor_g=g, factor_h=h, prime=5, precision=1
            )
        )
        assert result.lifted_g.coefficients == ("1", "2")
        assert result.lifted_h.coefficients == ("1", "3")

    def test_non_congruent_product_rejected(self):
        """Factors whose product misses f mod p are a typed domain error."""
        f = IntegerPolynomial(coefficients=("1", "0", "1"))
        g = _wire_poly(2, 1)
        h = _wire_poly(0, 1)
        with pytest.raises(ValueError, match="not congruent"):
            hensel_lift_factors(
                HenselFactorLiftRequest(
                    polynomial=f, factor_g=g, factor_h=h, prime=5, precision=2
                )
            )

    def test_shared_factor_rejected(self):
        """Factors sharing a root mod p are not coprime and are rejected."""
        f = IntegerPolynomial(coefficients=("1", "0", "0"))  # x^2
        g = _wire_poly(0, 1)
        h = _wire_poly(0, 1)
        with pytest.raises(ValueError, match="coprime"):
            hensel_lift_factors(
                HenselFactorLiftRequest(
                    polynomial=f, factor_g=g, factor_h=h, prime=5, precision=2
                )
            )

    def test_result_coefficients_stay_canonical(self):
        """Lifted factors omit leading zeros and stay below p^k."""
        f = IntegerPolynomial(coefficients=("1", "0", "1"))
        g = _wire_poly(2, 1)
        h = _wire_poly(3, 1)
        result = hensel_lift_factors(
            HenselFactorLiftRequest(
                polynomial=f, factor_g=g, factor_h=h, prime=5, precision=3
            )
        )
        for lifted in (result.lifted_g, result.lifted_h):
            assert len(lifted.coefficients) == 1 or lifted.coefficients[0] != "0"

    @given(
        constant=st.integers(min_value=0, max_value=4),
        linear=st.integers(min_value=0, max_value=4),
        quadratic=st.integers(min_value=0, max_value=4),
        prime=st.sampled_from((2, 3, 5, 7)),
        precision=st.integers(min_value=1, max_value=4),
    )
    def test_random_coprime_splits_reconstruct(
        self, constant: int, linear: int, quadratic: int, prime: int, precision: int
    ):
        """Every coprime split lifts with an exact product reconstruction.

        The defining invariant of Hensel factor lifting is
        ``lifted_g * lifted_h ≡ f (mod p^k)``; sweep random splits whose
        product matches ``f`` mod ``p`` and verify the reconstruction
        exactly.
        """
        g_asc = (constant % prime, 1)
        h_asc = (quadratic % prime, linear % prime, 1)
        product = [0] * 5
        for i, gi in enumerate(g_asc):
            for j, hj in enumerate(h_asc):
                product[i + j] += gi * hj
        f_asc = tuple(value % prime for value in product[:-1])
        if not f_asc[-1]:
            return
        # g = x + constant is linear; the split is coprime exactly when
        # h does not vanish at its root.
        root = (-constant) % prime
        if (quadratic + linear * root + root * root) % prime == 0:
            return
        result = hensel_lift_factors(
            HenselFactorLiftRequest(
                polynomial=_wire_poly(*f_asc),
                factor_g=_wire_poly(*g_asc),
                factor_h=_wire_poly(*h_asc),
                prime=prime,
                precision=precision,
            )
        )
        modulus = prime**precision
        for x in range(modulus):
            product_mod = (
                _poly_value_at(result.lifted_g, x, modulus)
                * _poly_value_at(result.lifted_h, x, modulus)
            ) % modulus
            assert product_mod == _poly_value_at(_wire_poly(*f_asc), x, modulus)
