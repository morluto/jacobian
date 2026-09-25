"""Typed degreewise congruence quotients of finite simplicial-set prefixes."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICES_PER_DEGREE,
    MAX_SIMPLICIAL_SET_DEGREE,
    FiniteTruncatedSimplicialSet,
)
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap

# Class IDs are labels: their magnitude is independent of the source simplex
# count and only their equality defines the relation. Keep them exactly
# interoperable as JSON numbers by bounding them to the JSON-safe integer
# envelope, rather than capping them by the 96-simplex carrier.
MAX_CLASS_ID = 2**53 - 1

ClassId = Annotated[
    StrictInt,
    Field(
        ge=0,
        le=MAX_CLASS_ID,
        description=(
            "A source-local equivalence class ID: a nonnegative JSON-safe "
            "integer label whose magnitude is independent of the simplex count."
        ),
    ),
]
DegreeClassIds = Annotated[
    tuple[ClassId, ...], Field(max_length=MAX_SIMPLICES_PER_DEGREE)
]


class SimplicialSetQuotientRequest(StrictModel):
    """Partition each degree of a finite prefix into proposed equivalence classes.

    Equal class IDs in a degree mean equivalent simplices. IDs have no meaning
    across degrees; the induced relation is checked against every visible face
    and degeneracy map before quotient tables are constructed.
    """

    simplicial_set: FiniteTruncatedSimplicialSet
    degree_class_ids: tuple[DegreeClassIds, ...] = Field(
        max_length=MAX_SIMPLICIAL_SET_DEGREE + 1,
        description=(
            "One class ID per simplex in each degree 0..N. Equality of IDs "
            "defines the degreewise equivalence relation."
        ),
    )

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if len(self.degree_class_ids) != self.simplicial_set.max_degree + 1:
            raise ValueError("degree_class_ids must cover every source degree")
        for degree, class_ids in enumerate(self.degree_class_ids):
            if len(class_ids) != len(self.simplicial_set.sets[degree]):
                raise ValueError(
                    f"degree_class_ids[{degree}] must align with the source simplex axis"
                )
        return self


class SimplicialSetQuotientResult(StrictModel):
    """A quotient prefix presented by its exact surjective projection map."""

    quotient_map: TruncatedSimplicialMap


__all__ = [
    "MAX_CLASS_ID",
    "SimplicialSetQuotientRequest",
    "SimplicialSetQuotientResult",
]
