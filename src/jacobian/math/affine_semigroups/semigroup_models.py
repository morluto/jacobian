from __future__ import annotations

from typing import Annotated

from pydantic import Field

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.affine_semigroups.semigroup import (
    MAX_AFFINE_GENERATORS,
    MAX_AFFINE_GRAPH_MOVES,
    AffineConfiguration,
    AffineFiber,
    AffineFiberGraph,
    AffineHilbertBasis,
    AffineMembershipResult,
    AffineSemigroupNormalization,
    PositiveAffineSemigroup,
    PositiveGradingResult,
)


class PositiveGradingRequest(StrictModel):
    configuration: AffineConfiguration


class AffineSemigroupRequest(StrictModel):
    configuration: AffineConfiguration
    grading: tuple[CanonicalRational, ...]


class AffineFiberRequest(StrictModel):
    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...] = Field(
        description=(
            "Target on the ambient row axis. After canonicalization, the "
            "product of target-derived coefficient ranges must fit the "
            "50,000-state exact fiber envelope."
        )
    )


class AffineMembershipRequest(AffineFiberRequest):
    pass


class AffineFiberGraphRequest(AffineFiberRequest):
    moves: tuple[
        Annotated[tuple[ExactInteger, ...], Field(max_length=MAX_AFFINE_GENERATORS)],
        ...,
    ] = Field(
        description=(
            "At most 16 nonzero integer kernel vectors on the generator axis. "
            "Each move z must satisfy A z = 0 exactly."
        ),
        max_length=MAX_AFFINE_GRAPH_MOVES,
    )


class AffineHilbertBasisRequest(StrictModel):
    configuration: AffineConfiguration = Field(
        description=(
            "Generators of a full-dimensional pointed cone in Z^2. The exact "
            "primitive-ray determinant must be at most 1,000."
        )
    )


class AffineSemigroupNormalizationRequest(StrictModel):
    semigroup: PositiveAffineSemigroup


__all__ = [
    "AffineFiber",
    "AffineFiberGraph",
    "AffineFiberGraphRequest",
    "AffineFiberRequest",
    "AffineHilbertBasis",
    "AffineHilbertBasisRequest",
    "AffineMembershipRequest",
    "AffineMembershipResult",
    "AffineSemigroupNormalization",
    "AffineSemigroupNormalizationRequest",
    "AffineSemigroupRequest",
    "PositiveAffineSemigroup",
    "PositiveGradingRequest",
    "PositiveGradingResult",
]
