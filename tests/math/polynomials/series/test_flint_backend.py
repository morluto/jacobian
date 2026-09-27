"""Differential and scale checks for the exact FLINT series kernels."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from jacobian._exact import CanonicalRational
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.math.polynomials.series import inverse, reversion
from jacobian.math.polynomials.series._models import TruncatedSeries


def _series(values: list[Fraction]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="x",
        truncation_order=len(values),
        coefficients=tuple(CanonicalRational.from_fraction(value) for value in values),
    )


def _oracle_inverse(values: list[Fraction]) -> list[Fraction]:
    result = [Fraction()] * len(values)
    result[0] = 1 / values[0]
    for degree in range(1, len(values)):
        result[degree] = -result[0] * sum(
            (values[index] * result[degree - index] for index in range(1, degree + 1)),
            Fraction(),
        )
    return result


def _oracle_compose(outer: list[Fraction], inner: list[Fraction]) -> list[Fraction]:
    order = len(outer)
    result = [Fraction()] * order
    power = [Fraction(1)] + [Fraction()] * (order - 1)
    for coefficient in outer:
        for degree, value in enumerate(power):
            result[degree] += coefficient * value
        next_power = [Fraction()] * order
        for degree in range(order):
            next_power[degree] = sum(
                (power[index] * inner[degree - index] for index in range(degree + 1)),
                Fraction(),
            )
        power = next_power
    return result


def _oracle_reversion(values: list[Fraction]) -> list[Fraction]:
    order = len(values)
    result = [Fraction()] * order
    result[1] = 1 / values[1]
    for degree in range(2, order):
        powers = [[Fraction()] * (degree + 1) for _ in range(degree + 1)]
        powers[0][0] = 1
        powers[1][:degree] = result[:degree]
        for exponent in range(2, degree + 1):
            powers[exponent] = [
                sum(
                    (
                        powers[exponent - 1][index] * powers[1][coefficient - index]
                        for index in range(coefficient + 1)
                    ),
                    Fraction(),
                )
                for coefficient in range(degree + 1)
            ]
        known = sum(
            (
                values[exponent] * powers[exponent][degree]
                for exponent in range(2, degree + 1)
            ),
            Fraction(),
        )
        result[degree] = -known / values[1]
    return result


@pytest.mark.parametrize("order", [64, 512])
def test_inverse_boundary_orders_are_exact(order: int) -> None:
    source = _series([Fraction(1), Fraction(1), *[Fraction()] * (order - 2)])
    result = inverse(source)

    assert len(result.result.coefficients) == order
    assert [value.as_fraction() for value in result.result.coefficients] == [
        (-1) ** index for index in range(order)
    ]
    assert len(result.residual_coefficients) == order
    assert all(value.num == 0 for value in result.residual_coefficients)


@pytest.mark.parametrize("order", [512])
def test_reversion_nonlinear_boundary_orders_are_exact(order: int) -> None:
    source = _series(
        [Fraction(0), Fraction(1), Fraction(1), *[Fraction()] * (order - 3)]
    )
    result = reversion(source)

    assert len(result.result.coefficients) == order
    assert [value.as_fraction() for value in result.result.coefficients] == [
        Fraction(0),
        *(
            Fraction((-1) ** (index - 1) * comb(2 * index - 2, index - 1), index)
            for index in range(1, order)
        ),
    ]
    assert len(result.left_residual) == order
    assert len(result.right_residual) == order
    assert all(
        value.num == 0 for value in (*result.left_residual, *result.right_residual)
    )


@pytest.mark.parametrize("order", [512])
def test_reversion_linear_boundary_orders_are_exact(order: int) -> None:
    source = _series([Fraction(0), Fraction(1), *[Fraction()] * (order - 2)])
    result = reversion(source)
    assert [value.as_fraction() for value in result.result.coefficients] == [
        Fraction(0),
        Fraction(1),
        *[Fraction()] * (order - 2),
    ]


@pytest.mark.parametrize(
    ("inverse_values", "reversion_values"),
    [
        (
            [Fraction(2), Fraction(1, 3), Fraction(-2), Fraction(5, 7)],
            [Fraction(0), Fraction(2), Fraction(1, 3), Fraction(-1, 2)],
        ),
        (
            [
                Fraction(-3, 2),
                Fraction(2),
                Fraction(1, 5),
                Fraction(-3, 7),
                Fraction(1),
            ],
            [
                Fraction(0),
                Fraction(-2),
                Fraction(3, 5),
                Fraction(1, 4),
                Fraction(-2, 3),
            ],
        ),
    ],
)
def test_small_orders_match_independent_python_oracles(
    inverse_values: list[Fraction], reversion_values: list[Fraction]
) -> None:
    inverse_result = inverse(_series(inverse_values))
    assert [value.as_fraction() for value in inverse_result.result.coefficients] == (
        _oracle_inverse(inverse_values)
    )

    reversion_result = reversion(_series(reversion_values))
    expected = _oracle_reversion(reversion_values)
    actual = [value.as_fraction() for value in reversion_result.result.coefficients]
    assert actual == expected
    assert _oracle_compose(reversion_values, expected) == [
        0,
        1,
        0,
        *[0] * (len(expected) - 3),
    ]
    assert _oracle_compose(expected, reversion_values) == [
        0,
        1,
        0,
        *[0] * (len(expected) - 3),
    ]


def test_series_backend_does_not_mutate_flint_global_cap() -> None:
    from flint import ctx

    original_cap = ctx.cap
    source = _series([Fraction(1), Fraction(1), *[Fraction()] * 62])
    inverse(source)
    reversion(_series([Fraction(0), Fraction(1), *[Fraction()] * 62]))
    assert ctx.cap == original_cap


def test_backend_failure_is_operational(monkeypatch: pytest.MonkeyPatch) -> None:
    from jacobian.math.polynomials.series import _flint

    monkeypatch.setattr(
        _flint,
        "_series_from_fractions",
        lambda coefficients, order: (_ for _ in ()).throw(RuntimeError("backend")),
    )
    with pytest.raises(OperationBackendError) as error:
        inverse(_series([Fraction(1), Fraction(1)]))
    assert error.value.reason == BackendFailureReason.INVALID_OUTPUT
