"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/friable_enumerate/test_friable_enumerate.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

FIVE_SMOOTH_THROUGH_20 = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20)


def _prime_factors(value: int) -> Iterator[int]:
    """Yield the distinct prime factors of one positive integer."""

    divisor = 2
    while divisor * divisor <= value:
        if value % divisor:
            divisor += 1
            continue
        yield divisor
        while value % divisor == 0:
            value //= divisor
        divisor += 1
    if value > 1:
        yield value


def _direct_enumerate(x: int, y: int) -> list[int]:
    """Brute-force oracle: factor every integer in 1..x."""

    if x == 0:
        return []
    if y <= 1:
        return [1] if x >= 1 else []
    result = []
    for n in range(1, x + 1):
        if all(p <= y for p in _prime_factors(n)):
            result.append(n)
    return result


def test_operation_is_discoverable_with_one_executable_example() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "integer.friable.enumerate"
    )
    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate_json(
        json.dumps(example.input), strict=True
    )
    result = operation.run(request)
    assert result.family.elements == FIVE_SMOOTH_THROUGH_20
