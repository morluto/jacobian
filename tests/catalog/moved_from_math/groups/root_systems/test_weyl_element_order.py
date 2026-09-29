"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/root_systems/test_weyl_element_order.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.groups.root_systems._models import (
    WeylElementRequest,
)
from jacobian.math.groups.root_systems._tools import TOOLS


def _matmul(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(len(left)))
            for j in range(len(right))
        )
        for i in range(len(left))
    )


def _a2_reflection(index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    # Matrices on the simple-root basis, derived directly from
    # s_i(alpha_j) = alpha_j - A_ij alpha_i.
    return (((-1, 1), (0, 1)), ((1, 0), (1, -1)))[index]


def _matrix_power_order(matrix: tuple[tuple[int, ...], ...], cap: int = 12) -> int:
    identity = tuple(
        tuple(int(i == j) for j in range(len(matrix))) for i in range(len(matrix))
    )
    power = identity
    for exponent in range(1, cap + 1):
        power = _matmul(power, matrix)
        if power == identity:
            return exponent
    raise AssertionError("oracle cap did not contain the finite order")


def test_manifest_example_invokes_published_order_operation() -> None:
    local = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "weyl_group.element.order.compute"
    )
    request = WeylElementRequest.model_validate_json(
        json.dumps(local.examples[0].input), strict=True
    )
    public = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "weyl_group.element.order.compute"
    )
    assert public.run(request).order == 3
