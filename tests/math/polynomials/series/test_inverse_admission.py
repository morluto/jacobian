"""Useful inverse envelopes checked by independent finite convolution."""

import json
from fractions import Fraction
from typing import Any, cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.polynomials.series import (
    divide,
    inverse,
    reversion,
    verify_divide,
    verify_inverse,
    verify_reversion,
)
from jacobian.math.polynomials.series._models import (
    MAX_TRUNCATION_ORDER,
    TruncatedSeries,
)
from jacobian.math.polynomials.series._tools import TOOLS


def _series(values: list[Fraction]) -> TruncatedSeries:
    return TruncatedSeries(
        variable="x",
        truncation_order=len(values),
        coefficients=tuple(CanonicalRational.from_fraction(value) for value in values),
    )


def _product(left: TruncatedSeries, right: TruncatedSeries) -> list[Fraction]:
    return [
        sum(
            (
                left.coefficients[i].as_fraction()
                * right.coefficients[k - i].as_fraction()
                for i in range(k + 1)
            ),
            Fraction(),
        )
        for k in range(left.truncation_order)
    ]


def test_serialized_residual_ledgers_are_claims_checked_by_consumers() -> None:
    source = _series([Fraction(1), Fraction(1), Fraction(0)])
    inverse_claim = inverse(source)
    decoded_inverse = type(inverse_claim).model_validate_json(
        inverse_claim.model_dump_json()
    )
    forged_inverse = decoded_inverse.model_copy(
        update={
            "residual_coefficients": (
                *decoded_inverse.residual_coefficients[:1],
                CanonicalRational.from_integer_ratio(1, 1),
                *decoded_inverse.residual_coefficients[2:],
            )
        }
    )
    assert not verify_inverse(forged_inverse)

    numerator = _series([Fraction(1), Fraction(2), Fraction(0)])
    division_claim = divide(numerator, source)
    decoded_division = type(division_claim).model_validate_json(
        division_claim.model_dump_json()
    )
    forged_division = decoded_division.model_copy(
        update={
            "residual_coefficients": (
                CanonicalRational.from_integer_ratio(1, 1),
                *decoded_division.residual_coefficients[1:],
            )
        }
    )
    assert not verify_divide(forged_division)

    reversion_claim = reversion(_series([Fraction(0), Fraction(1), Fraction(0)]))
    decoded_reversion = type(reversion_claim).model_validate_json(
        reversion_claim.model_dump_json()
    )
    forged_reversion = decoded_reversion.model_copy(
        update={
            "left_residual": (
                CanonicalRational.from_integer_ratio(1, 1),
                *decoded_reversion.left_residual[1:],
            )
        }
    )
    assert not verify_reversion(forged_reversion)


@pytest.mark.parametrize("order", [5, 6, 7, 64, MAX_TRUNCATION_ORDER])
@pytest.mark.parametrize("linear", [0, 1])
def test_constant_and_geometric_inverse(order: int, linear: int) -> None:
    source = _series([Fraction(1), Fraction(linear), *[Fraction()] * (order - 2)])
    result = inverse(source)
    assert [c.as_fraction() for c in result.result.coefficients] == [
        (-linear) ** i for i in range(order)
    ]
    assert _product(source, result.result) == [1, *[0] * (order - 1)]
    assert all(c.as_fraction() == 0 for c in result.residual_coefficients)


def test_dense_rational_inverse_and_division() -> None:
    order = 64
    source = _series([Fraction((-1) ** i * (i + 1), 6) for i in range(order)])
    numerator = _series([Fraction(i % 5 - 2, 7) for i in range(order)])
    result = inverse(source)
    assert _product(source, result.result) == [1, *[0] * (order - 1)]
    quotient = divide(numerator, source)
    assert _product(source, quotient.quotient) == [
        c.as_fraction() for c in numerator.coefficients
    ]
    assert all(c.as_fraction() == 0 for c in quotient.residual_coefficients)


@pytest.mark.parametrize(
    "values",
    [
        [Fraction(1, 2**700), *[Fraction(1)] * 7],
        [Fraction(-2, 3), Fraction(3, 7), Fraction(-5, 11), *[Fraction(1, 13)] * 13],
    ],
)
def test_nonunit_shared_denominators(values: list[Fraction]) -> None:
    source = _series(values)
    result = inverse(source)
    assert _product(source, result.result) == [1, *[0] * (len(values) - 1)]
    quotient = divide(source, source)
    assert [c.as_fraction() for c in quotient.quotient.coefficients] == [
        1,
        *[0] * (len(values) - 1),
    ]


@pytest.mark.parametrize("operation", ["inverse", "divide"])
def test_strict_json_execution_at_order_boundary(operation: str) -> None:
    source = _series(
        [Fraction(1), Fraction(1), *[Fraction()] * (MAX_TRUNCATION_ORDER - 2)]
    )
    tool = cast(
        MathTool[Any, StrictModel],
        next(
            item
            for item in TOOLS
            if item.operation_id == f"formal_series.rational.{operation}.compute"
        ),
    )
    if operation == "inverse":
        payload = source.model_dump_json()
        expected = inverse(source).model_dump()
    else:
        payload = json.dumps(
            {
                "left": source.model_dump(mode="json"),
                "right": source.model_dump(mode="json"),
            }
        )
        quotient = divide(source, source)
        expected = quotient.model_dump()
        assert [c.as_fraction() for c in quotient.quotient.coefficients] == [
            1,
            *[0] * (MAX_TRUNCATION_ORDER - 1),
        ]
    assert (
        tool.run(tool.request_type.model_validate_json(payload)).model_dump()
        == expected
    )


def test_growth_rejection_then_small_inverse_recovers() -> None:
    source = _series([Fraction(1), Fraction(10**255), *[Fraction()] * 18])
    with pytest.raises(OperationDomainValidationError) as error:
        inverse(source)
    assert (
        error.value.errors()[0]["type"]
        == "formal_power_series.inverse_coefficient_growth"
    )
    small = _series([Fraction(1), Fraction(1), *[Fraction()] * 4])
    assert _product(small, inverse(small).result) == [1, 0, 0, 0, 0, 0]


def test_matching_nonzero_residual_ledgers_do_not_prove_claims() -> None:
    source = _series([Fraction(1), Fraction(1), Fraction(0)])
    inverse_claim = inverse(source)
    decoded_inverse = type(inverse_claim).model_validate_json(
        inverse_claim.model_dump_json()
    )
    forged_inverse = decoded_inverse.model_copy(
        update={
            "result": decoded_inverse.result.model_copy(
                update={"coefficients": (CanonicalRational(num=0, den=1),) * 3}
            ),
            "residual_coefficients": (
                CanonicalRational(num=-1, den=1),
                CanonicalRational(num=0, den=1),
                CanonicalRational(num=0, den=1),
            ),
        }
    )
    assert not verify_inverse(forged_inverse)

    numerator = _series([Fraction(1), Fraction(2), Fraction(0)])
    division_claim = divide(numerator, source)
    decoded_division = type(division_claim).model_validate_json(
        division_claim.model_dump_json()
    )
    forged_division = decoded_division.model_copy(
        update={
            "quotient": decoded_division.quotient.model_copy(
                update={"coefficients": (CanonicalRational(num=0, den=1),) * 3}
            ),
            "residual_coefficients": (
                CanonicalRational(num=-1, den=1),
                CanonicalRational(num=-2, den=1),
                CanonicalRational(num=0, den=1),
            ),
        }
    )
    assert not verify_divide(forged_division)

    reversion_claim = reversion(_series([Fraction(0), Fraction(1), Fraction(0)]))
    decoded_reversion = type(reversion_claim).model_validate_json(
        reversion_claim.model_dump_json()
    )
    nonidentity = (
        CanonicalRational(num=0, den=1),
        CanonicalRational(num=-1, den=1),
        CanonicalRational(num=0, den=1),
    )
    forged_reversion = decoded_reversion.model_copy(
        update={
            "result": decoded_reversion.result.model_copy(
                update={"coefficients": (CanonicalRational(num=0, den=1),) * 3}
            ),
            "left_residual": nonidentity,
            "right_residual": nonidentity,
        }
    )
    assert not verify_reversion(forged_reversion)
