"""Contract tests for rational coordinate tensors and retained chart loci."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_RANK,
    RationalCoordinateTensor,
)
from jacobian.math.polynomials.values import RationalFunction


def _sparse(*terms: tuple[int, tuple[int, ...]]) -> dict[str, Any]:
    return {
        "terms": [
            {
                "coefficient": {"num": str(coefficient), "den": "1"},
                "exponents": list(exponents),
            }
            for coefficient, exponents in terms
        ]
    }


def _function(
    variables: tuple[str, ...],
    *numerator: tuple[int, tuple[int, ...]],
    denominator: tuple[tuple[int, tuple[int, ...]], ...] | None = None,
) -> RationalFunction:
    return RationalFunction.model_validate(
        {
            "variables": list(variables),
            "numerator": _sparse(*numerator),
            "denominator": _sparse(
                *((1, (0,) * len(variables)),) if denominator is None else denominator
            ),
        }
    )


def _tensor(
    variables: tuple[str, ...],
    variance: tuple[str, ...],
    components: tuple[RationalFunction, ...],
    *,
    guards: tuple[dict[str, Any], ...] = (),
) -> RationalCoordinateTensor:
    return RationalCoordinateTensor.model_validate(
        {
            "coordinate_axis": list(variables),
            "variance": list(variance),
            "components": [
                component.model_dump(mode="json") for component in components
            ],
            "retained_nonzero_denominators": list(guards),
        }
    )


def test_tensor_shape_preflight_closes_at_the_component_boundary() -> None:
    variables = ("x", "y")
    zero = _function(variables)

    at_boundary = _tensor(
        variables,
        ("COVARIANT",) * MAX_RATIONAL_TENSOR_RANK,
        (zero,) * MAX_RATIONAL_TENSOR_COMPONENTS,
    )
    assert len(at_boundary.components) == MAX_RATIONAL_TENSOR_COMPONENTS

    with pytest.raises(ValidationError) as error:
        RationalCoordinateTensor.model_validate(
            {
                "coordinate_axis": ["x", "y"],
                "variance": ["COVARIANT"] * (MAX_RATIONAL_TENSOR_RANK + 1),
                "components": [{}] * (MAX_RATIONAL_TENSOR_COMPONENTS * 2),
                "retained_nonzero_denominators": [],
            }
        )
    assert error.value.errors()[0]["type"] == (
        "differential_geometry.tensor_component_budget"
    )


def test_tensor_rejects_a_missing_or_noncanonical_locus_guard() -> None:
    variables = ("x",)
    value = _function(
        variables,
        (1, (0,)),
        denominator=((1, (1,)),),
    )
    with pytest.raises(ValidationError) as missing:
        _tensor(variables, (), (value,))
    assert missing.value.errors()[0]["type"] == (
        "differential_geometry.tensor_locus_missing_denominator"
    )

    with pytest.raises(ValidationError) as nonmonic:
        _tensor(
            variables,
            (),
            (value,),
            guards=(_sparse((2, (1,))),),
        )
    assert nonmonic.value.errors()[0]["type"] == (
        "differential_geometry.tensor_locus_guard_monic"
    )

    with pytest.raises(ValidationError) as over_budget:
        _tensor(
            variables,
            (),
            (_function(variables, (1, (0,))),),
            guards=(_sparse((1, (65,))),),
        )
    assert over_budget.value.errors()[0]["type"] == (
        "differential_geometry.tensor_locus_guard_budget"
    )


def test_tensor_component_coefficient_height_closes_at_128_digits() -> None:
    variables = ("x",)
    at_boundary = _function(variables, (10**127, (0,)))
    assert _tensor(variables, (), (at_boundary,)).components == (at_boundary,)

    with pytest.raises(ValidationError, match="128-digit bound"):
        _function(variables, (10**128, (0,)))
