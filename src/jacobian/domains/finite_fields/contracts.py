"""Typed requests for atomic finite-field operations."""

from typing import Self

from pydantic import model_validator

from jacobian.contracts.base import ContractModel
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


class RestrictScalarsRequest(ContractModel):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint


class LinearMapRankRequest(ContractModel):
    direction: ProjectivePoint
    linear_map: FiniteLinearMap

    @model_validator(mode="after")
    def require_shared_prime_field(self) -> Self:
        if self.direction.presentation.characteristic != self.linear_map.matrix.prime:
            raise ValueError(
                "direction characteristic must match the linear-map matrix prime"
            )
        return self


class ProjectiveLineRequest(ContractModel):
    presentation: FiniteFieldPresentation
    axis: Axis


class DirectionRankLedgerRequest(ContractModel):
    subspace: FiniteDimensionalSubspace
    directions: ProjectiveLine


class OrbitDistributionRequest(ContractModel):
    ledger: DirectionRankLedger


class FiniteMapTableRequest(ContractModel):
    polynomial_map: FinitePolynomialMap


class FiberPartitionRequest(ContractModel):
    table: FiniteMapTable


class CollisionRequest(ContractModel):
    table: FiniteMapTable


class PermutationRequest(ContractModel):
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
