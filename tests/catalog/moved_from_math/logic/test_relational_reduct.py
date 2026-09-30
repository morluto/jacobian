"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/logic/test_relational_reduct.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
)

OPERATION_ID = "relational_structure.reduct.compute"


def _structure(
    signature: tuple[tuple[str, int], ...],
    tables: tuple[tuple[tuple[int, ...], ...], ...],
    carrier_size: int,
) -> FiniteRelationalStructure:
    from jacobian.math.logic.relational_structures import FiniteRelationSymbol

    return FiniteRelationalStructure(
        carrier_size=carrier_size,
        signature=tuple(
            FiniteRelationSymbol(symbol_id=symbol_id, arity=arity)
            for symbol_id, arity in signature
        ),
        relation_tables=tables,
    )


def test_operation_is_published_in_the_catalog() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids


TWO_SYMBOL = _structure(
    (("E", 2), ("P", 1)),
    (((0, 1),), ((1,),)),
    2,
)
