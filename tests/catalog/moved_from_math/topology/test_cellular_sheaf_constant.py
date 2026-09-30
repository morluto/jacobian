"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_cellular_sheaf_constant.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
)
from jacobian.math.topology.cellular_sheaves._models import (
    SheafCohomologyResult,
    SheafScalar,
)
from jacobian.math.topology.cellular_sheaves.constants._tools import TOOLS

OPERATION_ID = "cellular_sheaf.constant.compute"


def _bettis(result: SheafCohomologyResult) -> tuple[int, ...]:
    return tuple(group.betti_number for group in result.groups)


def _rank_over_qq(matrix: tuple[tuple[SheafScalar, ...], ...]) -> int:
    rows = [
        [
            Fraction(value.num, value.den)
            if isinstance(value, CanonicalRational)
            else Fraction(value)
            for value in row
        ]
        for row in matrix
    ]
    rank = 0
    column_count = len(rows[0]) if rows else 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or rows[row][column] == 0:
                continue
            multiplier = rows[row][column]
            rows[row] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank], strict=True)
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def test_constant_sheaf_manifest_publishes_one_canonical_result() -> None:
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
    result = tool.run(
        tool.request_type.model_validate_json(
            json.dumps(tool.examples[0].input), strict=True
        )
    )
    assert isinstance(result, FiniteCellularSheaf)
    assert len(TOOLS) == 1


_INTERVAL = canonical_complex(("a", "b"), (("a", "b"),))
_CIRCLE = canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c")))
_TRIANGLE = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
_POINT = canonical_complex(("x",), (("x",),))
