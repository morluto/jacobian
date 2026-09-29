"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/powerful_enumerate/test_powerful_enumerate.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations


def test_operation_is_discoverable_with_one_executable_example() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "integer.powerful.enumerate"
    )
    assert len(operation.examples) == 1
    example = operation.examples[0]
    request = operation.request_type.model_validate(example.input)
    result = operation.run(request)
    assert result.cutoff == 100
    assert 1 in result.family
    assert 4 in result.family
    assert 8 in result.family
    assert 9 in result.family
    assert 12 not in result.family
    assert result.count == len(result.family)
