"""Canonical real algebraic values and exact-order evidence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.real_algebraic import (
    RealAlgebraicOrderValue,
    RealAlgebraicValue,
    compare_real_algebraic,
    isolate_real_algebraic,
)


def _value(polynomial: tuple[str, ...], root: int) -> RealAlgebraicValue:
    return RealAlgebraicValue(polynomial=polynomial, real_root_index=root)


def test_minimal_polynomial_and_root_index_determine_one_value() -> None:
    positive_sqrt_two = _value(("1", "0", "-2"), 1)

    assert positive_sqrt_two.polynomial == ("1", "0", "-2")
    assert positive_sqrt_two.real_root_index == 1
    interval = isolate_real_algebraic(positive_sqrt_two)
    assert interval.lower.as_fraction() == 1
    assert interval.upper.as_fraction() == 2
    assert interval.interval_type == "OPEN"


def test_order_uses_one_exact_axis_for_distinct_minimal_polynomials() -> None:
    result = compare_real_algebraic(
        _value(("1", "0", "-2"), 1),
        _value(("2", "-3"), 0),
    )

    assert result.order == "LT"
    assert result.left_isolating_interval.upper.as_fraction() <= (
        result.right_isolating_interval.lower.as_fraction()
    )


def test_order_within_one_minimal_polynomial_uses_real_root_indices() -> None:
    negative_sqrt_two = _value(("1", "0", "-2"), 0)
    positive_sqrt_two = _value(("1", "0", "-2"), 1)

    assert compare_real_algebraic(negative_sqrt_two, positive_sqrt_two).order == "LT"
    assert compare_real_algebraic(positive_sqrt_two, positive_sqrt_two).order == "EQ"


@pytest.mark.parametrize(
    ("polynomial", "message"),
    [
        (("-1", "0", "2"), "positive leading"),
        (("2", "0", "-4"), "primitive"),
        (("1", "0", "-1"), "irreducible"),
    ],
)
def test_noncanonical_minimal_polynomials_are_rejected(
    polynomial: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _value(polynomial, 0)


def test_value_rejects_a_nonreal_or_missing_root() -> None:
    with pytest.raises(ValidationError, match="existing real root"):
        _value(("1", "0", "1"), 0)
    with pytest.raises(ValidationError, match="existing real root"):
        _value(("1", "0", "-2"), 2)


def test_degree_and_coefficient_boundaries_are_closed() -> None:
    degree_eight = _value(("1", "0", "0", "0", "0", "0", "0", "0", "-2"), 1)
    thousand_digit_leading = _value(("1" + "0" * 999, "1"), 0)

    assert len(degree_eight.polynomial) == 9
    assert len(thousand_digit_leading.polynomial[0]) == 1_000
    with pytest.raises(ValidationError):
        _value(("1",) + ("0",) * 8 + ("-2",), 1)
    with pytest.raises(ValidationError, match="1000-digit bound"):
        _value(("1" + "0" * 1_000, "1"), 0)


def test_order_result_rejects_forged_conclusion_or_evidence() -> None:
    result = compare_real_algebraic(
        _value(("1", "0", "-2"), 1),
        _value(("1", "0", "-3"), 1),
    )
    forged_order = result.model_dump()
    forged_order["order"] = "GT"
    with pytest.raises(ValidationError, match="order must match"):
        RealAlgebraicOrderValue.model_validate(forged_order)

    forged_interval = result.model_dump()
    forged_interval["left_isolating_interval"]["lower"] = {
        "num": "0",
        "den": "1",
    }
    with pytest.raises(ValidationError, match="left isolating interval"):
        RealAlgebraicOrderValue.model_validate(forged_interval)


def test_value_round_trips_without_backend_expressions() -> None:
    value = _value(("1", "0", "-2"), 1)

    assert RealAlgebraicValue.model_validate_json(value.model_dump_json()) == value
