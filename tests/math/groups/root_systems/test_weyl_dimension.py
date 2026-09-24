"""Weyl dimension formula fixtures, composition, and admission."""

from __future__ import annotations

from importlib import import_module

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    WeylDimensionRequest,
    WeylDimensionResult,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.weyl_dimension import weyl_dimension

A1 = ((2,),)
A2 = ((2, -1), (-1, 2))
B2 = ((2, -2), (-1, 2))
G2 = ((2, -3), (-1, 2))


@pytest.mark.parametrize(
    ("matrix", "weight", "expected"),
    (
        (A1, (3,), 4),
        (A2, (1, 0), 3),
        (A2, (1, 1), 8),
        (B2, (1, 0), 4),
        (G2, (1, 0), 7),
        (G2, (0, 1), 14),
    ),
)
def test_independent_small_weyl_dimensions(
    matrix: tuple[tuple[int, ...], ...], weight: tuple[int, ...], expected: int
) -> None:
    result = weyl_dimension(matrix, weight)
    assert result.dimension == expected
    assert (
        len(result.positive_root_factors)
        == {
            A1: 1,
            A2: 3,
            B2: 4,
            G2: 6,
        }[matrix]
    )
    numerator = 1
    denominator = 1
    for factor in result.positive_root_factors:
        numerator *= factor.numerator_pairing
        denominator *= factor.denominator_pairing
    assert numerator == expected * denominator


def test_reducible_cartan_matrix_multiplies_component_dimensions() -> None:
    result = weyl_dimension(((2, 0), (0, 2)), (2, 1))
    assert result.dimension == 3 * 2
    assert result.weight_axis == (0, 1)
    assert len(result.positive_root_factors) == 2


@pytest.mark.parametrize("weight", ((-1, 0), (1,), (1, 0, 0)))
def test_rejects_non_dominant_or_wrong_axis_native_weights(
    weight: tuple[int, ...],
) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        weyl_dimension(A2, weight)
    assert caught.value.errors()[0]["type"] == "root_system.invalid_dominant_weight"


def test_rejects_nonintegral_and_negative_weights_in_request() -> None:
    with pytest.raises(ValidationError):
        WeylDimensionRequest.model_validate({"matrix": A2, "highest_weight": [1, 0.5]})
    with pytest.raises(ValidationError):
        WeylDimensionRequest.model_validate({"matrix": A2, "highest_weight": [1, -1]})


def test_rejects_over_64_bit_weight_as_resource_admission() -> None:
    with pytest.raises(OperationResourceAdmissionError) as caught:
        weyl_dimension(A2, (1 << 64, 0))
    assert caught.value.errors()[0]["type"] == "root_system.weyl_dimension_bounds"


def test_output_bound_is_checked_before_root_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = import_module("jacobian.math.groups.root_systems.weyl_dimension")

    def unexpected(_matrix: object) -> object:
        raise AssertionError("root enumeration ran before output admission")

    monkeypatch.setattr(operation, "_MAX_DIMENSION_OUTPUT_BYTES", 1)
    monkeypatch.setattr(operation, "_positive_coroots_from_admitted", unexpected)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        operation.weyl_dimension(A2, (1, 0))
    assert caught.value.errors()[0]["type"] == "root_system.weyl_dimension_bounds"


def test_result_roundtrip_and_catalog_declaration() -> None:
    result = weyl_dimension(G2, (1, 0))
    assert WeylDimensionResult.model_validate_json(result.model_dump_json()) == result
    declarations = [
        tool
        for tool in TOOLS
        if tool.operation_id == "root_system.weyl_dimension.compute"
    ]
    assert len(declarations) == 1
    tool = declarations[0]
    assert tool.run(WeylDimensionRequest(matrix=G2, highest_weight=(1, 0))) == result
