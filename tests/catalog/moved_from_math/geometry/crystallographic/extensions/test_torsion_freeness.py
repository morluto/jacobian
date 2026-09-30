"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/crystallographic/extensions/test_torsion_freeness.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian.math.geometry.crystallographic.extensions._models import (
    FiniteLatticeExtension,
)


def _extension(
    *,
    table: list[list[int]],
    action: list[list[list[int]]],
    cocycle: list[list[list[int]]],
) -> FiniteLatticeExtension:
    return FiniteLatticeExtension.model_validate(
        {
            "multiplication_table": table,
            "action_matrices": action,
            "factor_set": cocycle,
        }
    )


def _klein_extension() -> FiniteLatticeExtension:
    return _extension(
        table=[[0, 1], [1, 0]],
        action=[[[1, 0], [0, 1]], [[1, 0], [0, -1]]],
        cocycle=[
            [[0, 0], [0, 0]],
            [[0, 0], [1, 0]],
        ],
    )


def test_catalog_example_is_an_executable_canonical_request() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "crystallographic.extension.torsion_freeness.decide"
    )
    import json

    source = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(source)
    assert result.torsion_free
