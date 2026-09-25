"""Exact root-length classes with a Euclidean B2 oracle."""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    RootLengthProfileResult,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import root_length_profile


def _b2_length(root: tuple[int, int]) -> Fraction:
    """Squared norm in alpha_0=(1,1), alpha_1=(-2,0)."""
    x, y = root[0] - 2 * root[1], root[0]
    return Fraction(x * x + y * y)


def test_b2_root_classes_match_independent_euclidean_realization() -> None:
    result = root_length_profile(CartanMatrix.model_validate(((2, -2), (-1, 2))))
    component = result.components[0]
    observed = {
        root: group.squared_length.as_fraction()
        for group in component.length_classes
        for root in group.roots
    }
    roots = ((0, 1), (1, 0), (1, 1), (2, 1))
    assert observed == {root: _b2_length(root) for root in roots}
    assert tuple(
        group.squared_length.as_fraction() for group in component.length_classes
    ) == (2, 4)
    assert tuple(
        group.squared_length_ratio_to_short.as_fraction()
        for group in component.length_classes
    ) == (1, 2)
    assert result.positive_roots == roots
    assert (
        RootLengthProfileResult.model_validate_json(result.model_dump_json()) == result
    )


def test_reducible_factors_keep_length_ratios_local() -> None:
    # A1 and B2 are disconnected. Their independent first-simple-root
    # normalizations produce different absolute values, but the classes and
    # ratios remain attached to their own simple-root components.
    matrix = ((2, 0, 0), (0, 2, -2), (0, -1, 2))
    result = root_length_profile(CartanMatrix.model_validate(matrix))
    assert tuple(component.simple_root_indices for component in result.components) == (
        (0,),
        (1, 2),
    )
    assert len(result.components[0].length_classes) == 1
    assert (
        result.components[0]
        .length_classes[0]
        .squared_length_ratio_to_short.as_fraction()
        == 1
    )
    assert tuple(
        group.squared_length_ratio_to_short.as_fraction()
        for group in result.components[1].length_classes
    ) == (1, 2)


def test_root_length_profile_is_published_and_example_is_exact() -> None:
    operation_id = "root_system.root_length_profile.compute"
    local = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    request = local.request_type.model_validate_json(
        json.dumps(local.examples[0].input), strict=True
    )
    result = local.run(request)
    assert (
        result.components[0]
        .length_classes[-1]
        .squared_length_ratio_to_short.as_fraction()
        == 2
    )
