"""Tests for arithmetic-function operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.arithmetic_functions import dirichlet_convolution
from jacobian.math.number_theory.arithmetic_functions._models import (
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rat(num: int, den: int = 1) -> dict[str, str]:
    return {"num": str(num), "den": str(den)}


def _vals(*vals: tuple[int, int]) -> list[dict[str, str]]:
    return [{"num": str(n), "den": str(d)} for n, d in vals]


def _frac(v: CanonicalRational) -> Fraction:
    return v.as_fraction()


def test_native_convolution_returns_canonical_values() -> None:
    values = tuple(CanonicalRational.from_fraction(value) for value in (1, 2, 3))
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

    def test_constant_one_convolution_gives_divisor_count(self) -> None:
        """1 * 1 = tau where 1 is the constant-one function and tau is the
        divisor-count function."""
        result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate(
                {
                    "f": _vals((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
                    "g": _vals((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
                }
            )
        )
        # tau: 1, 2, 2, 3, 2, 4
        assert [int(v.as_fraction()) for v in result.values] == [1, 2, 2, 3, 2, 4]

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
        assert [v.as_fraction() for v in second.values] == [
            Fraction(3),
            Fraction(5),
            Fraction(7),
            Fraction(11),
        ]

    def test_inverse_flag_returned(self) -> None:
        result = compute_mobius_transform(
            MobiusTransformRequest.model_validate(
                {"values": _vals((1, 1), (1, 1)), "inverse": True}
            )
        )
        assert result.inverse is True

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


class TestLargerN:
    def test_convolution_of_constant_one_up_to_20(self) -> None:
        """1 * 1 = tau for n = 20; verify known values of tau."""
        n = 20
        ones = [_rat(1, 1)] * n
        result = compute_dirichlet_convolution(
            DirichletConvolutionRequest.model_validate({"f": ones, "g": ones})
        )
        # tau(1..20): 1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5, 2, 6, 2, 6
        expected = [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5, 2, 6, 2, 6]
        assert [int(v.as_fraction()) for v in result.values] == expected
