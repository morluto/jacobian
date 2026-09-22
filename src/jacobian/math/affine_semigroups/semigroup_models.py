from __future__ import annotations

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
    target: tuple[ExactInteger, ...]


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
