"""Typed contracts for finite relational carrier relabeling."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.relational_structures._models import FiniteCspInstance
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_CARRIER,
    FiniteRelationalStructure,
)


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"relational.relabeling.{reason}", message)


def _require_permutation(
    permutation: tuple[int, ...], carrier_size: int, field: str
) -> None:
    if len(permutation) != carrier_size:
        raise _error(
            f"{field}_axis", "the carrier map must contain one image per source element"
        )
    if any(not 0 <= label < carrier_size for label in permutation):
        raise _error(
            f"{field}_range", "every carrier image must belong to the target carrier"
        )
    if len(set(permutation)) != carrier_size:
        raise _error(f"{field}_bijection", "a carrier relabeling must be bijective")


class RelationalCarrierRelabelingRequest(StrictModel):
    """Relabel the canonical carrier while retaining its relation signature."""

    source: FiniteRelationalStructure
    old_to_new: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "One new carrier label per old carrier element, in increasing old "
            "label order: old_to_new[i] is the new label of old element i. "
            "The entries must form a full permutation of 0..carrier_size-1, "
            "so the map is a total bijection on the source carrier."
        ),
    )

    @model_validator(mode="after")
    def require_bijection(self) -> Self:
        _require_permutation(self.old_to_new, self.source.carrier_size, "old_to_new")
        return self


class RelationalCarrierRelabeling(StrictModel):
    """A structure and its coordinate transport to a relabeled carrier.

    The maps are structural bijections. Relation preservation is established by
    the operation that constructs this value; consumers that accept a
    caller-authored serialized result must validate that relation claim under
    their own admitted work bound.
    """

    source: FiniteRelationalStructure
    target: FiniteRelationalStructure
    old_to_new: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "The applied carrier bijection: old_to_new[i] is the new label of "
            "old element i, in increasing old label order."
        ),
    )
    new_to_old: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "The exact inverse of old_to_new: new_to_old[j] is the old label "
            "mapped to new element j."
        ),
    )

    @model_validator(mode="after")
    def require_structure_and_maps(self) -> Self:
        if (
            self.source.carrier_size != self.target.carrier_size
            or self.source.signature != self.target.signature
        ):
            raise _error(
                "structure_binding",
                "carrier relabeling preserves carrier size and exact relation signature",
            )
        size = self.source.carrier_size
        _require_permutation(self.old_to_new, size, "old_to_new")
        _require_permutation(self.new_to_old, size, "new_to_old")
        if any(self.new_to_old[new] != old for old, new in enumerate(self.old_to_new)):
            raise _error("inverse_map", "the two carrier maps must be exact inverses")
        return self


class CspTemplateCarrierRelabelingRequest(StrictModel):
    """Relabel a CSP template without changing variables or occurrences."""

    instance: FiniteCspInstance
    old_to_new: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "One new template-carrier label per old carrier element, in "
            "increasing old label order: old_to_new[i] is the new label of old "
            "template element i. The entries must form a full permutation of "
            "0..template.carrier_size-1, a total bijection on the template "
            "carrier."
        ),
    )

    @model_validator(mode="after")
    def require_bijection(self) -> Self:
        _require_permutation(
            self.old_to_new, self.instance.template.carrier_size, "old_to_new"
        )
        return self


class CspTemplateCarrierRelabeling(StrictModel):
    """A CSP instance with only its target carrier coordinates transported."""

    source: FiniteCspInstance
    target: FiniteCspInstance
    old_to_new: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "The applied template-carrier bijection: old_to_new[i] is the new "
            "label of old template element i, in increasing old label order."
        ),
    )
    new_to_old: tuple[StrictInt, ...] = Field(
        max_length=MAX_RELATIONAL_CARRIER,
        description=(
            "The exact inverse of old_to_new: new_to_old[j] is the old template "
            "label mapped to new element j."
        ),
    )

    @model_validator(mode="after")
    def require_instance_and_maps(self) -> Self:
        if (
            self.source.variable_count != self.target.variable_count
            or self.source.constraints != self.target.constraints
            or self.source.template.carrier_size != self.target.template.carrier_size
            or self.source.template.signature != self.target.template.signature
        ):
            raise _error(
                "csp_binding",
                "relabeling preserves variables, ordered constraints, carrier size, and signature",
            )
        size = self.source.template.carrier_size
        _require_permutation(self.old_to_new, size, "old_to_new")
        _require_permutation(self.new_to_old, size, "new_to_old")
        if any(self.new_to_old[new] != old for old, new in enumerate(self.old_to_new)):
            raise _error("inverse_map", "the two carrier maps must be exact inverses")
        return self


__all__ = [
    "CspTemplateCarrierRelabeling",
    "CspTemplateCarrierRelabelingRequest",
    "RelationalCarrierRelabeling",
    "RelationalCarrierRelabelingRequest",
]
