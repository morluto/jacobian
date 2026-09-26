"""Exact positive-root to coroot transforms."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    RootToCorootRequest,
    RootToCorootResult,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import (
    positive_coroots,
    root_to_coroot,
)


def _b2_geometry_oracle(root: tuple[int, int]) -> tuple[tuple[int, int], Fraction]:
    """Compute the coroot from explicit Euclidean B2 vectors.

    The simple roots are alpha_0=(1,1), alpha_1=(-2,0), giving squared
    lengths 2 and 4 and inner product -2. Coroot coordinates are solved in
    the explicit coroot basis alpha_0^vee=(1,1), alpha_1^vee=(-1,0).
    """
    x = root[0] - 2 * root[1]
    y = root[0]
    squared_length = Fraction(x * x + y * y)
    coroot_x = Fraction(2 * x, 1) / squared_length
    coroot_y = Fraction(2 * y, 1) / squared_length
    # u*(1,1) + v*(-1,0) = (coroot_x, coroot_y).
    return (int(coroot_y), int(coroot_y - coroot_x)), squared_length


def test_b2_coroots_match_independent_euclidean_realization() -> None:
    result = positive_coroots(CartanMatrix.model_validate(((2, -2), (-1, 2))))

    observed = {
        pair.root_coefficients: (
            pair.coroot_coefficients,
            pair.squared_length.as_fraction(),
        )
        for pair in result.positive_root_coroot_pairs
    }
    expected = {
        root: _b2_geometry_oracle(root)[0:2]
        for root in ((1, 0), (0, 1), (1, 1), (2, 1))
    }

    assert observed == expected
    assert result.datum.cartan_matrix.entries == ((2, -2), (-1, 2))


def test_simply_laced_a2_coroots_equal_roots() -> None:
    result = positive_coroots(CartanMatrix.model_validate(((2, -1), (-1, 2))))

    assert all(
        pair.root_coefficients == pair.coroot_coefficients
        and pair.squared_length.as_fraction() == 2
        for pair in result.positive_root_coroot_pairs
    )


def test_published_coroot_tool_is_declared() -> None:
    operation_ids = {tool.operation_id for tool in TOOLS}
    assert "root_system.coroots.compute" in operation_ids
    assert "root_system.root_to_coroot.compute" in operation_ids


def test_single_positive_root_conversion_matches_euclidean_oracle() -> None:
    matrix = CartanMatrix.model_validate(((2, -2), (-1, 2)))
    expected_coroot, expected_length = _b2_geometry_oracle((1, 1))
    result = root_to_coroot(matrix, (1, 1))

    assert result.datum.cartan_matrix == matrix
    assert result.pair.root_coefficients == (1, 1)
    assert result.pair.coroot_coefficients == expected_coroot == (1, 2)
    assert result.pair.squared_length.as_fraction() == expected_length == 2
    assert RootToCorootResult.model_validate_json(result.model_dump_json()) == result


def test_single_root_conversion_requires_positive_root_of_source_datum() -> None:
    matrix = CartanMatrix.model_validate(((2, -1), (-1, 2)))
    with pytest.raises(OperationDomainValidationError, match="positive root"):
        root_to_coroot(matrix, (2, 0))


def test_single_root_conversion_catalog_example_executes() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "root_system.root_to_coroot.compute"
    )
    request = RootToCorootRequest.model_validate_json(
        json.dumps(operation.examples[0].input), strict=True
    )
    result = operation.run(request)
    assert result.pair.coroot_coefficients == (1, 2)
    assert result.pair.squared_length.as_fraction() == 2
