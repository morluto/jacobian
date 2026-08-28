"""Pydantic wire contracts for poset closure, dual, and subposet operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.core._models import (
    MAX_POSET_ELEMENTS,
    ElementLabel,
    FinitePoset,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"poset.{reason}", message)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class LowerClosureRequest(StrictModel):
    """Compute the lower closure ↓S of a subset S in a finite poset."""

    poset: FinitePoset
    subset: tuple[ElementLabel, ...] = Field(
        min_length=1, max_length=MAX_POSET_ELEMENTS
    )

    @model_validator(mode="after")
    def require_subset_in_poset(self) -> Self:
        elements = set(self.poset.elements)
        for s in self.subset:
            if s not in elements:
                raise _validation_error(
                    "subset_in_carrier", f"subset element {s!r} is not in the poset"
                )
        return self


class UpperClosureRequest(StrictModel):
    """Compute the upper closure ↑S of a subset S in a finite poset."""

    poset: FinitePoset
    subset: tuple[ElementLabel, ...] = Field(
        min_length=1, max_length=MAX_POSET_ELEMENTS
    )

    @model_validator(mode="after")
    def require_subset_in_poset(self) -> Self:
        elements = set(self.poset.elements)
        for s in self.subset:
            if s not in elements:
                raise _validation_error(
                    "subset_in_carrier", f"subset element {s!r} is not in the poset"
                )
        return self


class DualPosetRequest(StrictModel):
    """Compute the dual (order-reversed) of a finite poset."""

    poset: FinitePoset


class InducedSubposetRequest(StrictModel):
    """Compute the induced subposet on a given element subset."""

    poset: FinitePoset
    subset: tuple[ElementLabel, ...] = Field(
        min_length=1, max_length=MAX_POSET_ELEMENTS
    )

    @model_validator(mode="after")
    def require_subset_in_poset(self) -> Self:
        elements = set(self.poset.elements)
        for s in self.subset:
            if s not in elements:
                raise _validation_error(
                    "subset_in_carrier", f"subset element {s!r} is not in the poset"
                )
        return self


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class LowerClosureResult(StrictModel):
    """The lower closure ↓S = {x : x <= s for some s in S}."""

    poset_digest: str
    subset: tuple[ElementLabel, ...]
    closure: tuple[ElementLabel, ...]
    is_ideal: bool = True


class UpperClosureResult(StrictModel):
    """The upper closure ↑S = {x : s <= x for some s in S}."""

    poset_digest: str
    subset: tuple[ElementLabel, ...]
    closure: tuple[ElementLabel, ...]
    is_filter: bool = True


class DualPosetResult(StrictModel):
    """The dual poset with reversed order."""

    poset: FinitePoset
    transport_map: tuple[ElementLabel, ...]


class InducedSubposetResult(StrictModel):
    """The subposet induced by restricting to a subset of elements."""

    subposet: FinitePoset
    old_to_new: tuple[ElementLabel, ...]
