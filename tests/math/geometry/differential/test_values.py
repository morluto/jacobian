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
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_VARIABLES, RationalFunction


def _sparse(*terms: tuple[int, tuple[int, ...]]) -> dict[str, Any]:
    return {
        "terms": [
            {
                "coefficient": {"num": (coefficient), "den": 1},
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
            "components": [component.model_dump() for component in components],
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


def test_tensor_component_count_is_rejected_before_sequence_traversal() -> None:
    class UntraversableComponents(list[dict[str, Any]]):
        def __iter__(self) -> Any:
            raise AssertionError("over-budget components must not be traversed")

    components = UntraversableComponents([{}] * (MAX_RATIONAL_TENSOR_COMPONENTS + 1))

    with pytest.raises(ValidationError) as error:
        RationalCoordinateTensor.model_validate(
            {
                "coordinate_axis": ["x"],
                "variance": [],
                "components": components,
                "retained_nonzero_denominators": [],
            }
        )

    assert error.value.errors()[0]["type"] == (
        "differential_geometry.tensor_component_budget"
    )


@pytest.mark.parametrize(
    ("field", "length", "error_type"),
    (
        (
            "coordinate_axis",
            MAX_POLYNOMIAL_VARIABLES + 1,
            "differential_geometry.tensor_coordinate_axis_budget",
        ),
        (
            "variance",
            MAX_RATIONAL_TENSOR_RANK + 1,
            "differential_geometry.tensor_rank_budget",
        ),
    ),
)
def test_tensor_axis_and_rank_caps_precede_sequence_traversal(
    field: str, length: int, error_type: str
) -> None:
    class UntraversableStrings(list[str]):
        def __iter__(self) -> Any:
            raise RuntimeError("over-budget shape metadata must not be traversed")

    payload: dict[str, Any] = {
        "coordinate_axis": ["x"],
        "variance": [],
        "components": [{}],
        "retained_nonzero_denominators": [],
    }
    payload[field] = UntraversableStrings(["x"] * length)

    with pytest.raises(ValidationError) as error:
        RationalCoordinateTensor.model_validate(payload)

    assert error.value.errors()[0]["type"] == error_type


def test_tensor_excessive_python_nesting_is_a_typed_validation_error() -> None:
    nested: Any = {}
    for _ in range(1_500):
        nested = [nested]

    with pytest.raises(ValidationError) as error:
        RationalCoordinateTensor.model_validate(
            {
                "coordinate_axis": ["x"],
                "variance": [],
                "components": [nested],
                "retained_nonzero_denominators": [],
            }
        )

    assert error.value.errors()[0]["type"] == (
        "differential_geometry.tensor_input_depth"
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
