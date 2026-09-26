"""Typed requests and values for finite simplicial subobject prefixes."""

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

DegreeSimplexIndices = Annotated[
    tuple[StrictInt, ...], Field(max_length=MAX_SIMPLICES_PER_DEGREE)
]


class SimplicialSubsetRequest(StrictModel):
    """Select a source-index family in each degree of one finite prefix.

    Each row is the strictly increasing list of ambient simplex indices to
    retain in that degree. Rows cover every degree, including empty families.
    """

    simplicial_set: FiniteTruncatedSimplicialSet
    degree_indices: tuple[DegreeSimplexIndices, ...] = Field(
        max_length=MAX_SIMPLICIAL_SET_DEGREE + 1,
        description=(
            "One strictly increasing list of ambient simplex indices per "
            "degree 0..max_degree; empty lists are permitted."
        ),
    )

    @model_validator(mode="after")
    def require_complete_degree_axes(self) -> Self:
        if len(self.degree_indices) != self.simplicial_set.max_degree + 1:
            raise ValueError("degree_indices must cover every source degree")
        for degree, indices in enumerate(self.degree_indices):
            if indices != tuple(sorted(set(indices))) or any(
                not 0 <= index < len(self.simplicial_set.sets[degree])
                for index in indices
            ):
                raise ValueError(
                    f"degree_indices[{degree}] must be increasing distinct source indices"
                )
        return self


class SimplicialSubsetPrefix(StrictModel):
    """A finite simplicial subobject presented by its exact inclusion map.

    The inclusion source is the selected prefix and its target is the exact
    ambient prefix. Decoding checks the degree axes, injective index rows, and
    retained labels. The producer checks face/degeneracy closure and naturality;
    consumers of caller-supplied serialized values recheck those relations at
    their own admitted operation boundary.
    """

    inclusion: TruncatedSimplicialMap

    @model_validator(mode="after")
    def require_inclusion_axes(self) -> Self:
        subset = self.inclusion.source
        ambient = self.inclusion.target
        if subset.max_degree != ambient.max_degree:
            raise ValueError("a finite subobject retains the complete ambient prefix")
        if len(self.inclusion.maps) != ambient.max_degree + 1:
            raise ValueError("inclusion maps must cover every ambient degree")
        for degree, index_map in enumerate(self.inclusion.maps):
            if index_map != tuple(sorted(set(index_map))):
                raise ValueError("subobject inclusions must be injective and ordered")
            expected_labels = tuple(ambient.sets[degree][index] for index in index_map)
            if subset.sets[degree] != expected_labels:
                raise ValueError("subset simplex labels must retain their ambient axes")
        return self


__all__ = ["SimplicialSubsetPrefix", "SimplicialSubsetRequest"]
