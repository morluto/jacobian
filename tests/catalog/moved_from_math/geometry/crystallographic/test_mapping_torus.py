"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/crystallographic/test_mapping_torus.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.crystallographic._models import (
    CrystallographicMappingTorusRequest,
)
from jacobian.math.geometry.crystallographic.operations import (
    mapping_torus_chain_complex,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.chain_complexes.operations import homology_groups
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    IntegralHomologyGroupValue,
)


def _matrix(rows: tuple[tuple[int, ...], ...]) -> IntegerMatrix:
    return IntegerMatrix(entries=rows)


def _integral_groups(
    complex_value: ChainComplexValue,
) -> tuple[IntegralHomologyGroupValue, ...]:
    groups = homology_groups(complex_value).homology_groups
    assert all(isinstance(group, IntegralHomologyGroupValue) for group in groups)
    return tuple(
        group for group in groups if isinstance(group, IntegralHomologyGroupValue)
    )


def test_request_schema_and_catalog_example_are_executable() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "crystallographic.mapping_torus.chain_complex.compute"
    )
    example = tool.examples[0]
    request = CrystallographicMappingTorusRequest.model_validate_json(
        json.dumps(example.input)
    )

    assert tool.run(request) == mapping_torus_chain_complex(_matrix(((-1,),)), 2)
    schema = CrystallographicMappingTorusRequest.model_json_schema()
    assert "A^m = I" in schema["properties"]["finite_order_exponent"]["description"]
