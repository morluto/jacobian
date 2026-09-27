"""Tests for arithmetic-function operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.arithmetic_functions import (
    dirichlet_convolution,
    dirichlet_inverse,
)
from jacobian.math.number_theory.arithmetic_functions._models import (
    _MAX_DIVISOR_PREFIX_LENGTH,
    DirichletConvolutionRequest,
    DirichletInverseRequest,
    MobiusTransformRequest,
    SummatoryFunctionRequest,
)
from jacobian.math.number_theory.arithmetic_functions._tools import (
    compute_dirichlet_convolution,
    compute_dirichlet_inverse,
    compute_mobius_transform,
    compute_summatory_function,
)
from jacobian.math.number_theory.arithmetic_functions.operations import (
    MAX_DIVISOR_INCIDENCES,
    _divisor_incidence_count,
    _divisor_incidences,
    _SlotLCMHeightSums,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rat(num: int, den: int = 1) -> dict[str, int]:
    return {"num": num, "den": den}


def _vals(*vals: tuple[int, int]) -> list[dict[str, int]]:
    return [{"num": n, "den": d} for n, d in vals]


def _frac(v: CanonicalRational) -> Fraction:
    return v.as_fraction()


def test_native_convolution_returns_canonical_values() -> None:
    values = tuple(
        CanonicalRational.from_fraction(Fraction(value)) for value in (1, 2, 3)
    )
    result = dirichlet_convolution(values, values)
    assert tuple(value.as_fraction() for value in result) == (
        Fraction(1),
        Fraction(4),
        Fraction(6),
    )


# ---------------------------------------------------------------------------
# Dirichlet convolution
# ---------------------------------------------------------------------------


class TestDirichletConvolution:
    def test_identity_convolution_returns_f(self) -> None:
        """f * 1 = f when g is the identity function."""
        result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate(
                {
                    "f": _vals((1, 1), (2, 1), (3, 1), (4, 1)),
                    "g": _vals((1, 1), (0, 1), (0, 1), (0, 1)),
                }
            )
        )
        assert result.length == 4
        assert [_frac(v) for v in result.values] == [
            Fraction(1),
            Fraction(2),
            Fraction(3),
            Fraction(4),
        ]

    def test_identity_convolution_with_constant_one(self) -> None:
        """id * 1 = sigma (sum of divisors): sigma(k) = sum_{d|k} d."""
        result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate(
                {
                    "f": _vals((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)),
                    "g": _vals((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
                }
            )
        )
        # sigma: 1, 3, 4, 7, 6, 12
        assert [int(v.as_fraction()) for v in result.values] == [1, 3, 4, 7, 6, 12]

    def test_single_element(self) -> None:
        result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate(
                {"f": [_rat(3, 1)], "g": [_rat(4, 1)]}
            )
        )
        assert result.length == 1
        assert result.values[0].as_fraction() == Fraction(12)

    def test_rational_values(self) -> None:
        """Convolution preserves exact rational values."""
        result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate(
                {
                    "f": [_rat(1, 2), _rat(1, 3), _rat(1, 4), _rat(1, 5)],
                    "g": [_rat(1, 1), _rat(1, 1), _rat(1, 1), _rat(1, 1)],
                }
            )
        )
        # h(1) = f(1)*g(1) = 1/2
        # h(2) = f(1)*g(2) + f(2)*g(1) = 1/2 + 1/3 = 5/6
        # h(3) = f(1)*g(3) + f(3)*g(1) = 1/2 + 1/4 = 3/4
        # h(4) = f(1)*g(4) + f(2)*g(2) + f(4)*g(1) = 1/2 + 1/3 + 1/5 = 31/30
        assert result.values[0].as_fraction() == Fraction(1, 2)
        assert result.values[1].as_fraction() == Fraction(5, 6)
        assert result.values[2].as_fraction() == Fraction(3, 4)
        assert result.values[3].as_fraction() == Fraction(31, 30)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DirichletConvolutionRequest.model_validate(
                {"f": [_rat(1)], "g": [_rat(1), _rat(2)]}
            )

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DirichletConvolutionRequest.model_validate({"f": [], "g": []})


# ---------------------------------------------------------------------------
# Möbius transform
# ---------------------------------------------------------------------------


class TestMobiusTransform:
    def test_constant_one_transform_gives_epsilon(self) -> None:
        """Möbius transform of the constant-one function is epsilon:
        F = 1, f(K) = sum_{d|K} mu(d) = 1 if K=1 else 0."""
        result = compute_mobius_transform(
            MobiusTransformRequest.model_validate(
                {"values": _vals((1, 1), (1, 1), (1, 1), (1, 1)), "inverse": False}
            )
        )
        assert result.inverse is False
        assert [int(v.as_fraction()) for v in result.values] == [1, 0, 0, 0]

    def test_divisor_count_transform_gives_identity(self) -> None:
        """Möbius transform of the divisor-count function tau gives the
        identity function 1: since tau = 1 * 1, the transform of tau gives
        1 back."""
        result = compute_mobius_transform(
            MobiusTransformRequest.model_validate(
                {
                    "values": _vals((1, 1), (2, 1), (2, 1), (3, 1), (2, 1), (4, 1)),
                }
            )
        )
        assert [int(v.as_fraction()) for v in result.values] == [
            1,
            1,
            1,
            1,
            1,
            1,
        ]

    def test_transform_is_involution(self) -> None:
        """Applying the Möbius transform twice returns the original.

        The forward Möbius transform f = mu * F is its own inverse: applying
        it twice returns the original function F.
        """
        original = _vals((3, 1), (5, 1), (7, 1), (11, 1))
        first = compute_mobius_transform(
            MobiusTransformRequest.model_validate(
                {"values": original, "inverse": False}
            )
        )
        second = compute_mobius_transform(
            MobiusTransformRequest.model_validate(
                {"values": list(first.values), "inverse": True}
            )
        )
        assert first.inverse is False
        assert second.inverse is True
        assert [v.as_fraction() for v in second.values] == [
            Fraction(3),
            Fraction(5),
            Fraction(7),
            Fraction(11),
        ]

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MobiusTransformRequest.model_validate({"values": []})


# ---------------------------------------------------------------------------
# Summatory function
# ---------------------------------------------------------------------------


class TestSummatoryFunction:
    def test_identity_summatory_gives_triangular(self) -> None:
        result = compute_summatory_function(
            SummatoryFunctionRequest.model_validate(
                {"values": _vals((1, 1), (2, 1), (3, 1), (4, 1))}
            )
        )
        # S(K) = 1, 3, 6, 10
        assert [int(v.as_fraction()) for v in result.values] == [1, 3, 6, 10]

    def test_constant_one_summatory(self) -> None:
        result = compute_summatory_function(
            SummatoryFunctionRequest.model_validate(
                {"values": _vals((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))}
            )
        )
        assert [int(v.as_fraction()) for v in result.values] == [1, 2, 3, 4, 5]

    def test_single_element(self) -> None:
        result = compute_summatory_function(
            SummatoryFunctionRequest.model_validate({"values": [_rat(7, 1)]})
        )
        assert result.length == 1
        assert result.values[0].as_fraction() == Fraction(7)

    def test_rational_values(self) -> None:
        result = compute_summatory_function(
            SummatoryFunctionRequest.model_validate(
                {"values": [_rat(1, 2), _rat(1, 3), _rat(1, 4)]}
            )
        )
        assert result.values[0].as_fraction() == Fraction(1, 2)
        assert result.values[1].as_fraction() == Fraction(5, 6)
        assert result.values[2].as_fraction() == Fraction(13, 12)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SummatoryFunctionRequest.model_validate({"values": []})


# ---------------------------------------------------------------------------
# Dirichlet inverse
# ---------------------------------------------------------------------------


class TestDirichletInverse:
    def test_inverse_of_constant_one_is_mobius(self) -> None:
        """The Dirichlet inverse of the constant-one function is the Möbius
        function mu."""
        result = compute_dirichlet_inverse(
            DirichletInverseRequest.model_validate(
                {"values": _vals((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1))}
            )
        )
        # mu: 1, -1, -1, 0, -1, 1
        assert [int(v.as_fraction()) for v in result.values] == [
            1,
            -1,
            -1,
            0,
            -1,
            1,
        ]

    def test_inverse_then_convolution_gives_epsilon(self) -> None:
        """f * g = epsilon when g is the Dirichlet inverse of f."""
        f_vals = _vals((2, 1), (3, 1), (5, 1), (7, 1))
        inv_result = compute_dirichlet_inverse(
            DirichletInverseRequest.model_validate({"values": f_vals})
        )
        conv_result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate(
                {"f": f_vals, "g": list(inv_result.values)}
            )
        )
        # epsilon: 1, 0, 0, 0
        assert conv_result.values[0].as_fraction() == Fraction(1)
        for i in range(1, 4):
            assert conv_result.values[i].as_fraction() == Fraction(0)

    def test_rational_inverse(self) -> None:
        """The Dirichlet inverse of f = id (with f(1) = 1) is mu."""
        result = compute_dirichlet_inverse(
            DirichletInverseRequest.model_validate(
                {"values": _vals((1, 1), (2, 1), (3, 1), (4, 1))}
            )
        )
        # g(1) = 1
        # g(2) = -(1/1)*(f(2)*g(1)) = -(2*1) = -2
        # g(3) = -(1/1)*(f(3)*g(1)) = -(3*1) = -3
        # g(4) = -(1/1)*(f(2)*g(2) + f(4)*g(1)) = -(2*(-2) + 4*1) = -(-4+4) = 0
        assert result.values[0].as_fraction() == Fraction(1)
        assert result.values[1].as_fraction() == Fraction(-2)
        assert result.values[2].as_fraction() == Fraction(-3)
        assert result.values[3].as_fraction() == Fraction(0)

    def test_zero_first_value_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            DirichletInverseRequest.model_validate(
                {"values": _vals((0, 1), (1, 1), (2, 1))}
            )
        assert exc_info.value.errors()[0]["type"] == "arithmetic_functions.zero_unit"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DirichletInverseRequest.model_validate({"values": []})


# ---------------------------------------------------------------------------
# Larger smoke test
# ---------------------------------------------------------------------------


def test_convolution_of_constant_one_up_to_20_gives_divisor_count() -> None:
    n = 20
    ones = [_rat(1, 1)] * n
    result = compute_dirichlet_convolution(
        DirichletConvolutionRequest.model_validate({"f": ones, "g": ones})
    )
    expected = [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5, 2, 6, 2, 6]
    assert [int(v.as_fraction()) for v in result.values] == expected


def test_divisor_prefix_bound_is_derived_from_incidence_work() -> None:
    assert _divisor_incidence_count(_MAX_DIVISOR_PREFIX_LENGTH) == 599_992
    assert _divisor_incidence_count(_MAX_DIVISOR_PREFIX_LENGTH + 1) == 600_032
    assert _divisor_incidence_count(_MAX_DIVISOR_PREFIX_LENGTH) <= (
        MAX_DIVISOR_INCIDENCES
    )
    assert _divisor_incidence_count(_MAX_DIVISOR_PREFIX_LENGTH + 1) > (
        MAX_DIVISOR_INCIDENCES
    )
    assert sum(1 for _ in _divisor_incidences(_MAX_DIVISOR_PREFIX_LENGTH)) == (599_992)


def test_mobius_max_prefix_admission_handles_many_denominators() -> None:
    """A shared prefix, two large primes, and many small primes fit the bound."""
    length = _MAX_DIVISOR_PREFIX_LENGTH
    first_prime = 1_000_000_007
    second_prime = 1_000_000_009
    small_primes: list[int] = []
    candidate = 2
    while len(small_primes) < 48:
        if all(candidate % prime != 0 for prime in small_primes):
            small_primes.append(candidate)
        candidate += 1

    common = first_prime * second_prime
    for prime in small_primes:
        common *= prime
    assert len(str(max(small_primes))) == 3
    assert len(str(common)) >= 80

    values = [CanonicalRational(num=1, den=first_prime)] * length
    values[-49:-1] = [CanonicalRational(num=1, den=prime) for prime in small_primes]
    values[-1] = CanonicalRational(num=1, den=second_prime)
    result = compute_mobius_transform(MobiusTransformRequest(values=tuple(values)))

    assert len(result.values) == length
    assert result.values[0] == CanonicalRational(num=1, den=first_prime)
    assert all(
        entry.num == 0 and entry.den == 1 for entry in result.values[1 : length - 49]
    )
    assert result.values[-1].as_fraction() == (
        Fraction(1, second_prime) - Fraction(1, first_prime)
    )
    assert all(common % int(entry.den) == 0 for entry in result.values)
    assert max(len(str(value.den)) for value in result.values) <= len(str(common))


def test_mobius_admission_accounts_for_lifting_carries() -> None:
    values = (
        CanonicalRational(num=-97, den=10),
        CanonicalRational(num=70, den=99),
    )
    result = compute_mobius_transform(MobiusTransformRequest(values=values))
    assert result.values[0].as_fraction() == Fraction(-97, 10)
    assert result.values[1].as_fraction() == Fraction(70, 99) - Fraction(-97, 10)


def test_mobius_transform_returns_exact_values_with_a_huge_denominator_lcm() -> None:
    huge_denominator = 10**4299 + 1
    values = (
        CanonicalRational(num=1, den=2),
        CanonicalRational(num=1, den=huge_denominator),
    )
    result = compute_mobius_transform(MobiusTransformRequest(values=values))
    assert result.values[0].as_fraction() == Fraction(1, 2)
    assert result.values[1].as_fraction() == (
        Fraction(1, huge_denominator) - Fraction(1, 2)
    )
    assert result.values[1].den.bit_length() > 14_000


def test_mobius_mixed_two_prime_denominators_match_the_defining_sum() -> None:
    first_prime = 7
    second_prime = 11
    values = tuple(
        CanonicalRational(
            num=1,
            den=first_prime if index % 2 == 0 else second_prime,
        )
        for index in range(12)
    )
    result = compute_mobius_transform(MobiusTransformRequest(values=values))
    mobius = (1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0)
    fractions = tuple(value.as_fraction() for value in values)
    for index in range(1, len(values) + 1):
        expected = sum(
            (
                mobius[divisor - 1] * fractions[index // divisor - 1]
                for divisor in range(1, index + 1)
                if index % divisor == 0
            ),
            start=Fraction(0),
        )
        assert result.values[index - 1].as_fraction() == expected


def test_slot_lcm_height_sums_require_concrete_length_checked_slots() -> None:
    with pytest.raises(ValueError):
        _SlotLCMHeightSums(2, (1,))
    with pytest.raises(ValueError):
        _SlotLCMHeightSums(1, (0,))


def test_convolution_max_prefix_preserves_shared_and_slot_denominators() -> None:
    length = _MAX_DIVISOR_PREFIX_LENGTH
    shared_denominator = 1_000_000_007
    exceptional_denominator = 10**100 + 1
    values = [CanonicalRational(num=1, den=shared_denominator) for _ in range(length)]
    values[-1] = CanonicalRational(num=1, den=exceptional_denominator)
    result = dirichlet_convolution(tuple(values), tuple(values))
    common_denominator_square = shared_denominator**2
    assert len(result) == length
    assert result[0] == CanonicalRational(num=1, den=common_denominator_square)
    assert all(entry.den == common_denominator_square for entry in result[:-1])
    assert result[5].as_fraction() == Fraction(4, common_denominator_square)
    assert result[49_999].as_fraction() == Fraction(30, common_denominator_square)
    assert result[-1].as_fraction() == Fraction(
        2, shared_denominator * exceptional_denominator
    )


def test_constant_one_inverse_admits_widened_prefix() -> None:
    ones = (CanonicalRational(num=1, den=1),) * _MAX_DIVISOR_PREFIX_LENGTH
    result = dirichlet_inverse(ones)

    assert len(result) == _MAX_DIVISOR_PREFIX_LENGTH
    assert result[0] == CanonicalRational(num=1, den=1)
    assert result[1].as_fraction() == Fraction(-1)
    assert all(
        entry.as_fraction() in {Fraction(-1), Fraction(0), Fraction(1)}
        for entry in result
    )
    source = tuple(
        CanonicalRational.from_fraction(Fraction(index, index + 1))
        for index in range(1, 33)
    )
    convolution = dirichlet_convolution(source, source)
    inverse = compute_dirichlet_inverse(DirichletInverseRequest(values=source))
    forward = compute_mobius_transform(MobiusTransformRequest(values=source))
    restored = compute_mobius_transform(
        MobiusTransformRequest(values=forward.values, inverse=True)
    )

    for index in range(1, len(source) + 1):
        divisors = tuple(
            divisor for divisor in range(1, index + 1) if index % divisor == 0
        )
        assert convolution[index - 1].as_fraction() == sum(
            (
                source[divisor - 1].as_fraction()
                * source[index // divisor - 1].as_fraction()
                for divisor in divisors
            ),
            start=Fraction(0),
        )
        identity_value = sum(
            (
                source[divisor - 1].as_fraction()
                * inverse.values[index // divisor - 1].as_fraction()
                for divisor in divisors
            ),
            start=Fraction(0),
        )
        assert identity_value == (Fraction(1) if index == 1 else Fraction(0))
    assert restored.values == source
