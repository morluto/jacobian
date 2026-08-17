"""Typed requests for atomic finite-field operations."""

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_fields import (
    Axis,
    DirectionRankLedger,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    FinitePolynomialMap,
    ProjectiveLine,
    ProjectivePoint,
)


class RestrictScalarsRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint


class LinearMapRankRequest(StrictModel):
    direction: ProjectivePoint
    linear_map: FiniteLinearMap

    @model_validator(mode="after")
    def require_shared_prime_field(self) -> Self:
        if self.direction.presentation.characteristic != self.linear_map.matrix.prime:
            raise ValueError(
                "direction characteristic must match the linear-map matrix prime"
            )
        return self


class ProjectiveLineRequest(StrictModel):
    presentation: FiniteFieldPresentation
    axis: Axis

    @model_validator(mode="after")
    def require_two_coordinate_axis(self) -> Self:
        if len(self.axis.labels) != 2:
            raise ValueError(
                "projective-line enumeration requires a two-coordinate axis"
            )
        return self


class DirectionRankLedgerRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    directions: ProjectiveLine


class OrbitDistributionRequest(StrictModel):
    ledger: DirectionRankLedger


class FiniteMapTableRequest(StrictModel):
    polynomial_map: FinitePolynomialMap


class FiberPartitionRequest(StrictModel):
    table: FiniteMapTable


class CollisionRequest(StrictModel):
    table: FiniteMapTable


class PermutationRequest(StrictModel):
    table: FiniteMapTable


__all__ = [
    "CollisionRequest",
    "DirectionRankLedgerRequest",
    "FiberPartitionRequest",
    "FiniteMapTableRequest",
    "LinearMapRankRequest",
    "OrbitDistributionRequest",
    "PermutationRequest",
    "ProjectiveLineRequest",
    "RestrictScalarsRequest",
]
