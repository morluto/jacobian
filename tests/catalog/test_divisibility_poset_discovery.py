"""Catalog discovery check for the divisibility-poset operations."""

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
