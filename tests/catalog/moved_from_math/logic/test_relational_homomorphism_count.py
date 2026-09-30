"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/logic/test_relational_homomorphism_count.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)

OPERATION_ID = "relational.homomorphism.count.compute"


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


def _directed_edge() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, (((0, 1),),))


def _two_clique() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, (((0, 1), (1, 0)),))


def test_operation_is_published_in_the_catalog() -> None:
    operation_ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in operation_ids


_EDGE = (FiniteRelationSymbol(symbol_id="E", arity=2),)
_NULLARY = (FiniteRelationSymbol(symbol_id="N", arity=0),)
