"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/number_theory/divisibility_poset/test_divisibility_poset.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    FinitePoset,
)
from jacobian.math.number_theory._divisibility_poset import (
    compute_divisibility_poset,
)
from jacobian.math.number_theory._divisibility_poset_models import (
    MAX_DIVISIBILITY_SET_SIZE,
    DivisibilityPosetRequest,
)


def _compute(elements: list[int]) -> FinitePoset:
    request = DivisibilityPosetRequest.model_validate(
        {"values": {"elements": elements}}
    )
    return compute_divisibility_poset(request)


def test_catalog_discovery() -> None:
    ids = [tool.operation_id for tool in BUILTIN_TOOLS]
    assert "integer.divisibility_poset.compute" in ids
    assert "number_theory.divisibility_poset.compute" not in ids


MAX_ADMITTED_ELEMENTS = min(MAX_DIVISIBILITY_SET_SIZE, MAX_POSET_ELEMENTS)
