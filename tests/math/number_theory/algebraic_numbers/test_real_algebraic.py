"""Canonical real algebraic values and exact-order evidence."""

from __future__ import annotations

import cProfile
from types import CodeType

import pytest
from pydantic import ValidationError
from tests.math.number_theory.algebraic_numbers._real_algebraic_support import (
    real_algebraic_validation_error,
)

from jacobian.math.number_theory.algebraic_numbers.real import (
    RealAlgebraicValue,
    compare_real_algebraic,
    isolate_real_algebraic,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._models import (
    AlgebraicCompareRequest,
    UnivariatePolynomialRequest,
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


def test_order_within_one_degree_sixteen_polynomial_uses_root_indices() -> None:
    negative_root = _value(("1",) + ("0",) * 15 + ("-2",), 0)
    positive_root = _value(("1",) + ("0",) * 15 + ("-2",), 1)

    assert compare_real_algebraic(negative_root, positive_root).order == "LT"


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
    with real_algebraic_validation_error():
        _value(("1", "0", "1"), 0)
    with real_algebraic_validation_error():
        _value(("1", "0", "-2"), 2)


def test_degree_and_coefficient_boundaries_are_closed() -> None:
    degree_sixteen = _value(("1",) + ("0",) * 15 + ("-2",), 1)
    thousand_digit_leading = _value(("1" + "0" * 999, "1"), 0)

    assert len(degree_sixteen.polynomial) == 17
    assert len(thousand_digit_leading.polynomial[0]) == 1_000
    with pytest.raises(ValidationError):
        _value(("1",) + ("0",) * 16 + ("-2",), 1)
    with real_algebraic_validation_error():
        _value(("1" + "0" * 1_000, "1"), 0)


def test_value_round_trips_without_backend_expressions() -> None:
    value = _value(("1", "0", "-2"), 1)

    assert RealAlgebraicValue.model_validate_json(value.model_dump_json()) == value


def test_pairwise_comparison_retains_its_degree_eight_work_envelope() -> None:
    degree_sixteen = _value(("1",) + ("0",) * 15 + ("-2",), 1)

    with pytest.raises(ValueError, match="degree at most 8"):
        compare_real_algebraic(degree_sixteen, _value(("1", "-1"), 0))
    with pytest.raises(ValidationError, match="degree at most 8"):
        AlgebraicCompareRequest(
            left=degree_sixteen,
            right=_value(("1", "-1"), 0),
        )


def test_comparison_preflights_raw_degree_before_algebraic_recognition() -> None:
    degree_sixteen = {
        "polynomial": ["1", *("0" for _ in range(15)), "-2"],
        "real_root_index": 1,
    }
    payload = {
        "left": degree_sixteen,
        "right": {
            "polynomial": ["1", *("0" for _ in range(15)), "-3"],
            "real_root_index": 1,
        },
    }
    profiler = cProfile.Profile()

    with pytest.raises(ValidationError, match="degree at most 8"):
        profiler.runcall(AlgebraicCompareRequest.model_validate, payload)

    assert (
        AlgebraicCompareRequest.model_json_schema()["properties"]["left"]["properties"][
            "polynomial"
        ]["maxItems"]
        == 17
    )
    assert not any(
        isinstance(entry.code, CodeType)
        and "/sympy/" in entry.code.co_filename.replace("\\", "/")
        for entry in profiler.getstats()
    )


def test_root_isolation_retains_its_degree_eight_work_envelope() -> None:
    coefficients = [
        {"num": "1" if index in {0, 9} else "0", "den": "1"} for index in range(10)
    ]

    with pytest.raises(ValidationError, match="degree at most 8"):
        UnivariatePolynomialRequest.model_validate(
            {"coefficients_descending": coefficients}
        )
