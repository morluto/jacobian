"""Tests for integer multiplicative normal-form operations (issue #1893)."""

from __future__ import annotations

import pytest

from jacobian.math.number_theory._normal_form_operations import (
    k_free_decomposition,
    normalize_positive_quadratic_radical,
    perfect_power_profile,
    squarefree_decomposition,
)
from jacobian.math.number_theory._normal_forms import (
    KFreeDecompositionRequest,
    PerfectPowerProfileRequest,
    QuadraticRadicalNormalizeRequest,
    SquarefreeDecompositionRequest,
)

# ---------------------------------------------------------------------------
# Perfect-power profile tests
# ---------------------------------------------------------------------------


class TestPerfectPowerProfile:
    def test_zero(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="0"))
        assert r.classification == "ZERO"
        assert r.base is None
        assert r.exponent is None

    def test_positive_unit(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="1"))
        assert r.classification == "POSITIVE_UNIT"

    def test_negative_unit(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="-1"))
        assert r.classification == "NEGATIVE_UNIT"

    def test_prime(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="7"))
        assert r.classification == "NONUNIT"
        assert r.base == "7"
        assert r.exponent == 1
        assert not r.is_nontrivial_perfect_power

    def test_64(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="64"))
        assert r.classification == "NONUNIT"
        assert r.base == "2"
        assert r.exponent == 6
        assert r.is_nontrivial_perfect_power

    def test_negative_64(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="-64"))
        assert r.classification == "NONUNIT"
        assert r.base == "-4"
        assert r.exponent == 3
        assert r.is_nontrivial_perfect_power

    def test_negative_16(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="-16"))
        assert r.classification == "NONUNIT"
        assert r.base == "-16"
        assert r.exponent == 1
        assert not r.is_nontrivial_perfect_power

    def test_72(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="72"))
        assert r.classification == "NONUNIT"
        assert r.base == "72"
        assert r.exponent == 1

    def test_4096(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="4096"))
        assert r.classification == "NONUNIT"
        assert r.base == "2"
        assert r.exponent == 12

    def test_729(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="729"))
        assert r.classification == "NONUNIT"
        assert r.base == "3"
        assert r.exponent == 6

    def test_negative_729(self) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value="-729"))
        assert r.classification == "NONUNIT"
        assert r.base == "-9"
        assert r.exponent == 3

    @pytest.mark.parametrize(
        "n", ["0", "1", "-1", "64", "-64", "72", "-16", "4096", "729", "-729"]
    )
    def test_reconstruction(self, n: str) -> None:
        r = perfect_power_profile(PerfectPowerProfileRequest(value=n))
        if r.classification == "NONUNIT":
            assert int(r.base) ** r.exponent == int(n)


# ---------------------------------------------------------------------------
# K-free decomposition tests
# ---------------------------------------------------------------------------


class TestKFreeDecomposition:
    def test_zero(self) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value="0", k=2))
        assert r.classification == "ZERO"
        assert r.extracted_base is None
        assert r.k_free_cofactor is None

    def test_72_k2(self) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value="72", k=2))
        assert r.classification == "NONZERO"
        assert int(r.extracted_base) ** 2 * int(r.k_free_cofactor) == 72

    def test_72_k3(self) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value="72", k=3))
        assert r.classification == "NONZERO"
        assert int(r.extracted_base) ** 3 * int(r.k_free_cofactor) == 72

    def test_negative_k2(self) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value="-72", k=2))
        assert r.classification == "NONZERO"
        assert int(r.extracted_base) ** 2 * int(r.k_free_cofactor) == -72
        assert int(r.k_free_cofactor) < 0

    def test_already_k_free(self) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value="6", k=2))
        assert r.extracted_base == "1"
        assert r.k_free_cofactor == "6"

    def test_k4(self) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value="16", k=4))
        assert r.extracted_base == "2"
        assert r.k_free_cofactor == "1"

    @pytest.mark.parametrize(
        "n,k", [("72", 2), ("72", 3), ("72", 4), ("-72", 2), ("6", 2), ("16", 4)]
    )
    def test_reconstruction(self, n: str, k: int) -> None:
        r = k_free_decomposition(KFreeDecompositionRequest(value=n, k=k))
        if r.classification == "NONZERO":
            assert int(r.extracted_base) ** k * int(r.k_free_cofactor) == int(n)

    @pytest.mark.parametrize("n,k", [("72", 2), ("72", 3), ("72", 4)])
    def test_cofactor_is_k_free(self, n: str, k: int) -> None:
        from sympy import factorint

        r = k_free_decomposition(KFreeDecompositionRequest(value=n, k=k))
        assert r.classification == "NONZERO"
        c = abs(int(r.k_free_cofactor))
        if c > 0:
            for _, e in factorint(c).items():
                assert e < k, f"cofactor {c} has a prime with exponent {e} >= k={k}"


# ---------------------------------------------------------------------------
# Squarefree decomposition tests
# ---------------------------------------------------------------------------


class TestSquarefreeDecomposition:
    def test_zero(self) -> None:
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value="0"))
        assert r.classification == "ZERO"

    def test_72(self) -> None:
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value="72"))
        assert r.classification == "NONZERO"
        assert r.square_factor == "6"
        assert r.signed_squarefree_part == "2"

    def test_negative(self) -> None:
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value="-72"))
        assert r.classification == "NONZERO"
        assert r.square_factor == "6"
        assert r.signed_squarefree_part == "-2"

    def test_squarefree_input(self) -> None:
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value="6"))
        assert r.square_factor == "1"
        assert r.signed_squarefree_part == "6"

    def test_perfect_square(self) -> None:
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value="144"))
        assert r.square_factor == "12"
        assert r.signed_squarefree_part == "1"

    @pytest.mark.parametrize("n", ["0", "1", "-1", "72", "-72", "6", "144", "100"])
    def test_reconstruction(self, n: str) -> None:
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value=n))
        if r.classification == "NONZERO":
            s = int(r.square_factor)
            d = int(r.signed_squarefree_part)
            assert s * s * d == int(n)

    def test_squarefree_part_vs_radical(self) -> None:
        """72 = 2^3 * 3^2 -> squarefree part = 2, not 6 (the radical)."""
        r = squarefree_decomposition(SquarefreeDecompositionRequest(value="72"))
        assert r.signed_squarefree_part == "2"
        # radical(72) = 2*3 = 6, which is different from the squarefree part
        assert r.signed_squarefree_part != "6"


# ---------------------------------------------------------------------------
# Quadratic radical normalization tests
# ---------------------------------------------------------------------------


class TestNormalizedQuadraticRadical:
    def test_zero(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="0")
        )
        assert r.classification == "ZERO"
        assert r.coefficient == "0"
        assert r.radicand == "1"

    def test_one(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="1")
        )
        assert r.classification == "RATIONAL_INTEGER"
        assert r.coefficient == "1"
        assert r.radicand == "1"

    def test_12(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="12")
        )
        assert r.coefficient == "2"
        assert r.radicand == "3"
        assert r.classification == "IRRATIONAL_QUADRATIC"

    def test_72(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="72")
        )
        assert r.coefficient == "6"
        assert r.radicand == "2"
        assert r.classification == "IRRATIONAL_QUADRATIC"

    def test_144(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="144")
        )
        assert r.coefficient == "12"
        assert r.radicand == "1"
        assert r.classification == "RATIONAL_INTEGER"

    def test_large_exact_square(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="10000")
        )
        assert r.coefficient == "100"
        assert r.radicand == "1"
        assert r.classification == "RATIONAL_INTEGER"

    @pytest.mark.parametrize("n", ["0", "1", "12", "72", "144", "10000", "7"])
    def test_reconstruction(self, n: str) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value=n)
        )
        s = int(r.coefficient)
        d = int(r.radicand)
        assert s * s * d == int(n)


# ---------------------------------------------------------------------------
# Cross-operation consistency tests
# ---------------------------------------------------------------------------


class TestCrossOperationConsistency:
    def test_squarefree_decomposition_matches_k2(self) -> None:
        n = "72"
        sq = squarefree_decomposition(SquarefreeDecompositionRequest(value=n))
        k2 = k_free_decomposition(KFreeDecompositionRequest(value=n, k=2))
        assert sq.square_factor == k2.extracted_base
        assert sq.signed_squarefree_part == k2.k_free_cofactor

    def test_radical_uses_squarefree(self) -> None:
        n = "72"
        sq = squarefree_decomposition(SquarefreeDecompositionRequest(value=n))
        rad = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value=n)
        )
        assert rad.coefficient == sq.square_factor
        assert rad.radicand == sq.signed_squarefree_part.lstrip("-")

    def test_perfect_square_radical_is_rational(self) -> None:
        r = normalize_positive_quadratic_radical(
            QuadraticRadicalNormalizeRequest(value="64")
        )
        assert r.classification == "RATIONAL_INTEGER"
        assert r.coefficient == "8"
        assert r.radicand == "1"
