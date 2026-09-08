from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


class EdgeOverlapMomentRow(StrictModel):
    edge_size_min: int = Field(ge=0)
    edge_size_max: int = Field(ge=0)
    intersection_size: int = Field(ge=0)
    pair_count: ExactInteger = Field(ge=0)
    union_size: int = Field(ge=0)
    covariance: CanonicalRational

class HypergraphEdgeCountMomentsRequest(StrictModel):
    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational
    @model_validator(mode="after")
    def validate_probability(self) -> Self:
        if not 0 <= self.retention_probability.as_fraction() <= 1:
            raise ValueError("retention_probability must be between 0 and 1")
        return self

class HypergraphEdgeCountMomentsResult(StrictModel):
    hypergraph: FiniteHypergraph
    retention_probability: CanonicalRational
    edge_count_expectation: CanonicalRational
    edge_count_second_moment: CanonicalRational
    edge_count_variance: CanonicalRational
    overlap_profile: tuple[EdgeOverlapMomentRow, ...]

__all__ = ["EdgeOverlapMomentRow", "HypergraphEdgeCountMomentsRequest", "HypergraphEdgeCountMomentsResult"]
