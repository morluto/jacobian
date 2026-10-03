"""Composition admission bounds exactly the powers consumed by its kernel."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import pytest

from jacobian._exact import CanonicalRational as Q
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.series import compose, operations, truncate
from jacobian.math.polynomials.series._models import TruncatedSeries


def series(
    values: Sequence[int | Fraction], order: int = 32, variable: str = "x"
) -> TruncatedSeries:
    return TruncatedSeries(
        variable=variable,
        truncation_order=order,
        coefficients=tuple(
            Q.from_fraction(Fraction(v))
            for v in [*values, *([0] * (order - len(values)))]
        ),
    )


def inner(order: int = 32) -> TruncatedSeries:
    return series([0, Fraction(10**200, 3), Fraction(10**200, 7)], order)


@pytest.mark.parametrize("order", [16, 32, 512])
@pytest.mark.parametrize("outer_values", [[], [5], [1, 1], [3, 2, -1]])
def test_mixed_denominator_composition_uses_only_active_outer_degree(
    order: int, outer_values: list[int]
) -> None:
    source = inner(order)
    outer = series(outer_values, order)
    result = compose(outer, source)
    a = Fraction(10**200, 3)
    b = Fraction(10**200, 7)
    if not outer_values:
        expected = []
    elif len(outer_values) == 1:
        expected = [Fraction(outer_values[0])]
    elif len(outer_values) == 2:
        expected = [Fraction(1), a, b]
    else:
        expected = [Fraction(3), 2 * a, 2 * b - a * a, -2 * a * b, -b * b]
    assert [v.as_fraction() for v in result.result.coefficients] == [
        *expected,
        *([Fraction()] * (order - len(expected))),
    ]
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result
    assert (
        truncate(decoded.result, 16).result.coefficients
        == result.result.coefficients[:16]
    )


def test_high_active_outer_degree_still_refuses_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = series([*([0] * 31), 1])

    def unexpected(*args: object) -> Any:
        pytest.fail("unadmitted power entered composition")

    monkeypatch.setattr(operations, "_compose_coefficients", unexpected)
    with pytest.raises(
        OperationResourceAdmissionError, match="composition coefficient growth"
    ):
        compose(outer, inner())


@pytest.mark.parametrize("values", [[], [7], [2, -3, 4]])
def test_zero_inner_retains_outer_constant(values: list[int]) -> None:
    result = compose(series(values), series([]))
    assert [v.as_fraction() for v in result.result.coefficients] == [
        values[0] if values else 0,
        *([0] * 31),
    ]


@pytest.mark.parametrize("values", [[], [7], [2, -3, 4]])
def test_constant_outer_does_not_bypass_inner_domain(values: list[int]) -> None:
    with pytest.raises(OperationDomainValidationError, match="zero constant"):
        compose(series(values), series([1, Fraction(2, 3), Fraction(3, 7)]))


@pytest.mark.parametrize("operand", ["outer", "inner"])
@pytest.mark.parametrize("bad", ["variable", "order", "length", "scalar"])
def test_shortcuts_still_validate_native_source_carriers(
    bad: str, operand: str
) -> None:
    source = inner()
    if bad == "variable":
        source = source.model_copy(update={"variable": "bad axis"})
    elif bad == "order":
        source = source.model_copy(update={"truncation_order": True})
    elif bad == "length":
        source = source.model_copy(update={"coefficients": source.coefficients[:3]})
    else:
        source = source.model_copy(
            update={
                "coefficients": (
                    Q.model_construct(num=0, den=2),
                    *source.coefficients[1:],
                )
            }
        )
    with pytest.raises(OperationDomainValidationError):
        compose(source, inner()) if operand == "outer" else compose(series([]), source)


def test_context_mismatch_is_not_erased_by_zero_outer() -> None:
    with pytest.raises(OperationDomainValidationError, match="same variable"):
        compose(series([], variable="q"), inner())
    with pytest.raises(OperationDomainValidationError, match="same truncation order"):
        compose(series([], order=16), inner())


def test_interior_zero_outer_coefficients_do_not_skip_needed_powers() -> None:
    # 2-G+3G^3: the absent quadratic term does not remove the needed G² step.
    g = series([0, Fraction(2, 3), Fraction(-5, 7)], 8)
    result = compose(series([2, -1, 0, 3], 8), g)
    a, b = Fraction(2, 3), Fraction(-5, 7)
    expected = [2, -a, -b, 3 * a**3, 9 * a * a * b, 9 * a * b * b, 3 * b**3, 0]
    assert [v.as_fraction() for v in result.result.coefficients] == expected


def test_zero_final_composition_does_not_erase_executed_intermediate_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # G has valuation two, so G^63 vanishes mod x^64; G^21 is still
    # executed on the way and its degree-42 coefficient has 4201 digits.
    outer = series([*([0] * 63), 1], 64)
    source = series([0, 0, Fraction(10**200, 3), Fraction(10**200, 7)], 64)

    def unexpected(*args: object) -> Any:
        pytest.fail("unadmitted executed intermediate entered composition")

    monkeypatch.setattr(operations, "_compose_coefficients", unexpected)
    with pytest.raises(OperationResourceAdmissionError):
        compose(outer, source)
