from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class EdgeOverlapMomentRow(StrictModel):
    edge_size_min: int = Field(ge=0, le=256)
    edge_size_max: int = Field(ge=0, le=256)
    intersection_size: int = Field(ge=0, le=256)
    pair_count: ExactInteger = Field(ge=1, le=71_994_000)
    union_size: int = Field(ge=0, le=256)
    covariance: CanonicalRational

    @model_validator(mode="after")
    def require_overlap_shape(self) -> Self:
        if not self.intersection_size <= self.edge_size_min <= self.edge_size_max:
            raise ValueError(
                "overlap sizes must satisfy intersection <= smaller <= larger"
            )
        if (
            self.union_size
            != self.edge_size_min + self.edge_size_max - self.intersection_size
        ):
            raise ValueError(
                "union size must equal the sum of edge sizes minus intersection"
            )
        return self


class HypergraphEdgeCountMomentsRequest(StrictModel):
    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational = Field(
        description=(
            "Exact independent vertex-retention probability in the closed "
            "interval [0, 1]. An edge survives with probability p^|e|."
        )
    )

    @model_validator(mode="after")
    def validate_probability(self) -> Self:
        probability = self.retention_probability
        if not 0 <= probability.num <= probability.den:
            raise ValueError("retention_probability must be between 0 and 1")
        return self


class HypergraphEdgeCountMomentsResult(StrictModel):
    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational
    edge_count_expectation: CanonicalRational
    edge_count_second_moment: CanonicalRational
    edge_count_variance: CanonicalRational
    overlap_profile: tuple[EdgeOverlapMomentRow, ...] = Field(max_length=65_536)


__all__ = [
    "EdgeOverlapMomentRow",
    "HypergraphEdgeCountMomentsRequest",
    "HypergraphEdgeCountMomentsResult",
]
