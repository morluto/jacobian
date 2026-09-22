from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    AffineFiber,
    AffineMembershipResult,
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


__all__ = [
    "AffineFiber",
    "AffineFiberRequest",
    "AffineMembershipRequest",
    "AffineMembershipResult",
    "AffineSemigroupRequest",
    "PositiveAffineSemigroup",
    "PositiveGradingRequest",
    "PositiveGradingResult",
]
