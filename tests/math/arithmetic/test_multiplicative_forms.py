"""Known-answer and adversarial tests for integer multiplicative normal forms."""

import pytest
from pydantic import ValidationError

from jacobian.math.arithmetic._multiplicative_forms import (
    IntegerKRequest,
    IntegerRequest,
    NonnegativeIntegerRequest,
)
from jacobian.math.arithmetic._multiplicative_operations import (
    compute_k_free_decomposition,
    compute_normalized_quadratic_radical,
    compute_perfect_power_profile,
    compute_squarefree_decomposition,
)


class TestPerfectPowerProfile:
    def test_zero(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="0"))
        assert result.kind == "ZERO"

    def test_positive_unit(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="1"))
        assert result.kind == "POSITIVE_UNIT"

    def test_negative_unit(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="-1"))
        assert result.kind == "NEGATIVE_UNIT"

    def test_64_is_2_6(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="64"))
        assert result.kind == "NONUNIT"
        assert result.base == "2"
        assert result.exponent == 6
        assert result.is_nontrivial_perfect_power is True

    def test_negative_64_is_neg4_3(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="-64"))
        assert result.kind == "NONUNIT"
        assert result.base == "-4"
        assert result.exponent == 3
        assert result.is_nontrivial_perfect_power is True

    def test_negative_16_exponent_1(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="-16"))
        assert result.kind == "NONUNIT"
        assert result.exponent == 1
        assert result.is_nontrivial_perfect_power is False

    def test_72_not_perfect_power(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="72"))
        assert result.kind == "NONUNIT"
        assert result.exponent == 1
        assert result.is_nontrivial_perfect_power is False

    def test_negative_729(self) -> None:
        # -729 = (-9)^3, since 729 = 3^6, gcd=6, odd part of 6 is 3
        result = compute_perfect_power_profile(IntegerRequest(value="-729"))
        assert result.kind == "NONUNIT"
        assert result.base == "-9"
        assert result.exponent == 3

    def test_reconstruction(self) -> None:
        for n in ["64", "-64", "729", "-729", "72", "-16", "2", "-1"]:
            result = compute_perfect_power_profile(IntegerRequest(value=n))
            if result.kind == "NONUNIT":
                base = int(result.base)
                exp = result.exponent
                assert base**exp == int(n), f"{base}^{exp} != {n}"

    def test_factor_rows(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="72"))
        assert result.kind == "NONUNIT"
        factors = {f.prime: f.power for f in result.factors}
        assert factors == {"2": 3, "3": 2}

    def test_4096_is_2_12(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="4096"))
        assert result.kind == "NONUNIT"
        assert result.base == "2"
        assert result.exponent == 12
        assert result.is_nontrivial_perfect_power is True

    def test_81_is_3_4(self) -> None:
        result = compute_perfect_power_profile(IntegerRequest(value="81"))
        assert result.kind == "NONUNIT"
        assert result.base == "3"
        assert result.exponent == 4


class TestKFreeDecomposition:
    def test_zero(self) -> None:
        result = compute_k_free_decomposition(IntegerKRequest(value="0", k=3))
        assert result.kind == "ZERO"

    def test_72_k3(self) -> None:
        # 72 = 2^3 * 3^2, k=3: a = 2^1 = 2, c = 3^2 = 9
        result = compute_k_free_decomposition(IntegerKRequest(value="72", k=3))
        assert result.kind == "NONUNIT"
        assert result.base == "2"
        assert result.cofactor == "9"
        assert result.reconstruction == "72"

    def test_negative_k3(self) -> None:
        result = compute_k_free_decomposition(IntegerKRequest(value="-72", k=3))
        assert result.kind == "NONUNIT"
        assert result.base == "2"
        assert result.cofactor == "-9"
        assert result.reconstruction == "-72"

    def test_already_k_free(self) -> None:
        # 30 = 2*3*5 is already 2-free (squarefree)
        result = compute_k_free_decomposition(IntegerKRequest(value="30", k=2))
        assert result.kind == "NONUNIT"
        assert result.base == "1"
        assert result.cofactor == "30"

    def test_k4(self) -> None:
        # 2^5 * 3^4, k=4: a = 2^1 * 3^1 = 6, c = 2^1 = 2
        # 2^5 = 5 divmod 4 = (1, 1), 3^4 = 4 divmod 4 = (1, 0)
        # a = 2*3 = 6, c = 2^1 = 2
        n = 2**5 * 3**4
        result = compute_k_free_decomposition(IntegerKRequest(value=str(n), k=4))
        assert result.kind == "NONUNIT"
        assert result.base == "6"
        assert result.cofactor == "2"
        assert result.reconstruction == str(n)

    def test_reconstruction(self) -> None:
        for n in ["72", "-72", "30", "-30", "1", "-1"]:
            for k in [2, 3, 4]:
                result = compute_k_free_decomposition(IntegerKRequest(value=n, k=k))
                if result.kind == "NONUNIT":
                    base = int(result.base)
                    cofactor = int(result.cofactor)
                    assert base**k * cofactor == int(n)

    def test_cofactor_exponents_below_k(self) -> None:
        result = compute_k_free_decomposition(IntegerKRequest(value="72", k=3))
        # cofactor = 9 = 3^2, exponent 2 < 3 ✓
        assert result.cofactor == "9"


class TestSquarefreeDecomposition:
    def test_zero(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="0"))
        assert result.kind == "ZERO"

    def test_72(self) -> None:
        # 72 = 2^3 * 3^2 = 6^2 * 2, so s=6, d=2
        result = compute_squarefree_decomposition(IntegerRequest(value="72"))
        assert result.kind == "NONUNIT"
        assert result.square_factor == "6"
        assert result.squarefree_part == "2"
        assert result.reconstruction == "72"

    def test_negative(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="-72"))
        assert result.kind == "NONUNIT"
        assert result.square_factor == "6"
        assert result.squarefree_part == "-2"

    def test_already_squarefree(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="30"))
        assert result.kind == "NONUNIT"
        assert result.square_factor == "1"
        assert result.squarefree_part == "30"

    def test_perfect_square(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="144"))
        assert result.kind == "NONUNIT"
        assert result.square_factor == "12"
        assert result.squarefree_part == "1"

    def test_distinct_from_radical(self) -> None:
        # 72 = 2^3 * 3^2, squarefree_part = 2, radical = 2*3 = 6
        result = compute_squarefree_decomposition(IntegerRequest(value="72"))
        assert result.squarefree_part == "2"

    def test_reconstruction(self) -> None:
        for n in ["72", "-72", "30", "-30", "144", "-144", "1", "-1"]:
            result = compute_squarefree_decomposition(IntegerRequest(value=n))
            if result.kind == "NONUNIT":
                s = int(result.square_factor)
                d = int(result.squarefree_part)
                assert s**2 * d == int(n)


class TestNormalizedQuadraticRadical:
    def test_zero(self) -> None:
        result = compute_normalized_quadratic_radical(
            NonnegativeIntegerRequest(value="0")
        )
        assert result.kind == "ZERO"
        assert result.coefficient == "0"
        assert result.radicand == "1"

    def test_one(self) -> None:
        result = compute_normalized_quadratic_radical(
            NonnegativeIntegerRequest(value="1")
        )
        assert result.kind == "RATIONAL_INTEGER"
        assert result.coefficient == "1"
        assert result.radicand == "1"

    def test_72(self) -> None:
        result = compute_normalized_quadratic_radical(
            NonnegativeIntegerRequest(value="72")
        )
        assert result.kind == "IRRATIONAL_QUADRATIC"
        assert result.coefficient == "6"
        assert result.radicand == "2"

    def test_12(self) -> None:
        result = compute_normalized_quadratic_radical(
            NonnegativeIntegerRequest(value="12")
        )
        assert result.kind == "IRRATIONAL_QUADRATIC"
        assert result.coefficient == "2"
        assert result.radicand == "3"

    def test_144(self) -> None:
        result = compute_normalized_quadratic_radical(
            NonnegativeIntegerRequest(value="144")
        )
        assert result.kind == "RATIONAL_INTEGER"
        assert result.coefficient == "12"
        assert result.radicand == "1"

    def test_large_perfect_square(self) -> None:
        n = str(10**100)
        result = compute_normalized_quadratic_radical(
            NonnegativeIntegerRequest(value=n)
        )
        assert result.kind == "RATIONAL_INTEGER"
        assert result.radicand == "1"
        assert int(result.coefficient) ** 2 == int(n)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="nonnegative"):
            NonnegativeIntegerRequest(value="-1")

    def test_reconstruction(self) -> None:
        for n in ["0", "1", "12", "72", "144"]:
            result = compute_normalized_quadratic_radical(
                NonnegativeIntegerRequest(value=n)
            )
            s = int(result.coefficient)
            d = int(result.radicand)
            assert s**2 * d == int(n)


class TestUnitHandling:
    """Tests for the UNIT variant in k-free and squarefree decompositions."""

    def test_k_free_unit_1(self) -> None:
        result = compute_k_free_decomposition(IntegerKRequest(value="1", k=3))
        assert result.kind == "UNIT"

    def test_k_free_unit_neg1(self) -> None:
        result = compute_k_free_decomposition(IntegerKRequest(value="-1", k=3))
        assert result.kind == "UNIT"

    def test_squarefree_unit_1(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="1"))
        assert result.kind == "UNIT"

    def test_squarefree_unit_neg1(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="-1"))
        assert result.kind == "UNIT"

    def test_k_free_zero_unchanged(self) -> None:
        result = compute_k_free_decomposition(IntegerKRequest(value="0", k=3))
        assert result.kind == "ZERO"

    def test_squarefree_zero_unchanged(self) -> None:
        result = compute_squarefree_decomposition(IntegerRequest(value="0"))
        assert result.kind == "ZERO"

    def test_perfect_power_zero_no_fields(self) -> None:
        """ZERO variant must not carry NONUNIT fields."""
        result = compute_perfect_power_profile(IntegerRequest(value="0"))
        assert result.kind == "ZERO"
        assert result.base is None
        assert result.exponent is None
        assert result.factors == ()
        assert result.is_nontrivial_perfect_power is False
