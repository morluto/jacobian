"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/combinatorics/designs/incidence_structures/test_incidence_trade.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian.math.combinatorics.designs.incidence_structures import (
    IncidenceStructure,
)

_TRADE_POINTS = ("1", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "20")
_REMOVED_BLOCKS = (
    ("1",),
    ("13",),
    ("20",),
    ("5", "7"),
    ("5", "8"),
    ("6", "10"),
    ("6", "11"),
    ("1", "3", "5"),
    ("1", "4", "6"),
    ("3", "4", "9"),
    ("5", "6", "9"),
    ("7", "8", "13"),
    ("10", "11", "20"),
)
_INSERTED_BLOCKS = (
    ("1", "5"),
    ("1", "6"),
    ("5", "6"),
    ("7", "13"),
    ("8", "13"),
    ("10", "20"),
    ("11", "20"),
    ("1", "3", "4"),
    ("3", "5", "9"),
    ("4", "6", "9"),
    ("5", "7", "8"),
    ("6", "10", "11"),
)


def _family(
    blocks: tuple[tuple[str, ...], ...],
    prefix: str,
    *,
    points: tuple[str, ...] = _TRADE_POINTS,
) -> IncidenceStructure:
    return IncidenceStructure(
        points=points,
        block_ids=tuple(f"{prefix}{index}" for index in range(len(blocks))),
        blocks=blocks,
    )


def _long_id_family(prefix: str, filler: str, id_length: int) -> IncidenceStructure:
    return IncidenceStructure(
        points=("a",),
        block_ids=tuple(
            f"{prefix}{index}-" + filler * (id_length - len(f"{prefix}{index}-"))
            for index in range(100)
        ),
        blocks=((),) * 100,
    )


def test_trade_operation_is_discoverable_and_example_executes() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool for tool in BUILTIN_TOOLS if tool.operation_id == "incidence.trade.check"
    )
    example = operation.examples[0]
    result = operation.run(operation.request_type.model_validate(example.input))
    assert result.positive_moments_equal
    assert result.zeroth_difference == 1
