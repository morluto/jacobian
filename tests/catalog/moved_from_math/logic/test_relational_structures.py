"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/logic/test_relational_structures.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

import json

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismStatus,
    check_homomorphism,
)
from jacobian.math.logic.relational_structures._models import (
    HomomorphismCheckRequest,
)

OPERATION_ID = "relational.homomorphism.check"


def _structure(
    carrier_size: int,
    signature: tuple[FiniteRelationSymbol, ...],
    tables: tuple[tuple[tuple[int, ...], ...], ...],
) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=carrier_size, signature=signature, relation_tables=tables
    )


def _three_cycle() -> FiniteRelationalStructure:
    return _structure(3, _EDGE, (((0, 1), (1, 2), (2, 0)),))


def _directed_triangle() -> FiniteRelationalStructure:
    return _structure(
        3,
        _EDGE,
        (((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),),
    )


def _loop_point() -> FiniteRelationalStructure:
    return _structure(1, _EDGE, (((0, 0),),))


def _edgeless_two() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, ((),))


def _catalog_tool():
    return next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)


def _wire_payload(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: tuple[int, ...],
) -> dict:
    return json.loads(
        HomomorphismCheckRequest(
            source=source, target=target, carrier_map=carrier_map
        ).model_dump_json()
    )


def test_native_and_catalog_paths_agree() -> None:
    tool = _catalog_tool()
    payload = _wire_payload(_three_cycle(), _directed_triangle(), (0, 1, 2))
    catalog_result = tool.run(tool.request_type.model_validate(payload))
    native_result = check_homomorphism(_three_cycle(), _directed_triangle(), (0, 1, 2))
    assert catalog_result == native_result

    negative_payload = _wire_payload(_directed_triangle(), _three_cycle(), (0, 1, 2))
    catalog_negative = tool.run(tool.request_type.model_validate(negative_payload))
    assert catalog_negative == check_homomorphism(
        _directed_triangle(), _three_cycle(), (0, 1, 2)
    )
    assert catalog_negative.status is HomomorphismStatus.NOT_HOMOMORPHISM


def test_catalog_example_executes() -> None:
    tool = _catalog_tool()
    for example in tool.examples:
        result = tool.run(
            tool.request_type.model_validate(json.loads(json.dumps(example.input)))
        )
        assert result.status is HomomorphismStatus.HOMOMORPHISM


_EDGE = (FiniteRelationSymbol(symbol_id="E", arity=2),)
