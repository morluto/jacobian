"""Canonical formal-series contracts for rational generating functions."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._recurrence_models import (
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
)
from jacobian.math.combinatorics.operations import (
    rational_generating_function_coefficients,
    verify_rational_generating_function_coefficients,
)
from jacobian.math.polynomials.series._models import SeriesTruncateRequest
from jacobian.math.polynomials.series.operations import truncate

ONE = CanonicalRational(num="1", den="1")
MINUS_ONE = CanonicalRational(num="-1", den="1")


def _expand(
    denominator: tuple[CanonicalRational, ...], order: int
) -> RationalGeneratingFunctionCoefficientsResult:
    return rational_generating_function_coefficients(
        (ONE,), denominator, "ASCENDING_POWERS_OF_X", "0", order
    )


def test_degree_32_recurrence_returns_a_bound_canonical_series() -> None:
    order = 4_096
    denominator = (ONE, *((MINUS_ONE,) * 32))
    request = RationalGeneratingFunctionCoefficientsRequest(
        numerator=(ONE,),
        denominator=denominator,
        coefficient_convention="ASCENDING_POWERS_OF_X",
        expansion_point="0",
        truncation_order=order,
    )

    result = _expand(request.denominator, request.truncation_order)
    values = tuple(value.as_fraction() for value in result.series.coefficients)
    denominator_values = tuple(value.as_fraction() for value in denominator)

    assert len(values) == order
    assert values[-1] == sum(values[-33:-1])
    assert all(
        sum(
            denominator_values[offset] * values[degree - offset]
            for offset in range(min(degree, len(denominator_values) - 1) + 1)
        )
        == (1 if degree == 0 else 0)
        for degree in range(order)
    )
    assert result.series.variable == "x"
    assert result.series.truncation_order == order
    assert verify_rational_generating_function_coefficients(result)
    assert (
        RationalGeneratingFunctionCoefficientsResult.model_validate_json(
            encode_strict_json(result.model_dump(mode="json"))
        )
        == result
    )

    forged = result.model_dump(mode="json")
    forged["series"]["coefficients"][0] = {"num": "2", "den": "1"}
    decoded = RationalGeneratingFunctionCoefficientsResult.model_validate(forged)
    assert not verify_rational_generating_function_coefficients(decoded)


def test_canonical_series_composes_into_formal_series_truncation() -> None:
    result = rational_generating_function_coefficients(
        (ONE,),
        (ONE, MINUS_ONE),
        "ASCENDING_POWERS_OF_X",
        "0",
        6,
    )
    request = SeriesTruncateRequest.model_validate(
        {"series": result.series.model_dump(mode="json"), "target_order": 3}
    )

    truncated = truncate(request.series, request.target_order)

    assert tuple(value.as_fraction() for value in truncated.result.coefficients) == (
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )


def test_source_relation_survives_canonical_json_roundtrip() -> None:
    result = _expand((ONE, MINUS_ONE), 5)
    wire = json.loads(encode_strict_json(result.model_dump(mode="json")))
    decoded = RationalGeneratingFunctionCoefficientsResult.model_validate(wire)

    assert decoded.numerator == (ONE,)
    assert decoded.denominator == (ONE, MINUS_ONE)
    assert verify_rational_generating_function_coefficients(decoded)


def test_catalog_schema_publishes_the_canonical_series_example() -> None:
    descriptor = Catalog.open().inspect(
        "combinatorics.generating_function.coefficients.compute"
    )
    assert descriptor is not None
    assert set(descriptor.output_schema["properties"]) == {
        "numerator",
        "denominator",
        "coefficient_convention",
        "expansion_point",
        "truncation_order",
        "series",
    }
    example = descriptor.output_schema["examples"][0]
    assert example["series"]["variable"] == "x"
    assert example["series"]["truncation_order"] == 3


def test_denominator_degree_controls_the_recurrence_work_envelope() -> None:
    denominator = (ONE, *((MINUS_ONE,) * 32))

    with pytest.raises(OperationDomainValidationError, match="exact work bound"):
        _expand(denominator, 8_000)


def test_large_constant_prefix_is_admitted_by_recurrence_work() -> None:
    assert len(_expand((ONE,), 250_000).series.coefficients) == 250_000
