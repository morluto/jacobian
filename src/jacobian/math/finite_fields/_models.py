"""Typed requests for atomic finite-field operations."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.finite_fields.values import (
    _MAX_HOMOGENEOUS_DEGREE,
    Axis,
    DirectionRankLedger,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteMapTable,
    FinitePolynomialMap,
    PrimeFieldLinearAction,
    ProjectiveLine,
    ProjectivePoint,
)

_MAX_PROJECTIVE_POINTS = 4_096
_MAX_DIRECTION_RANK_WORK = 1_000_000


class HomogeneousFixedSubspaceRequest(StrictModel):
    """Compute one homogeneous fixed space for explicit action generators."""

    action: PrimeFieldLinearAction
    degree: StrictInt = Field(ge=0, le=_MAX_HOMOGENEOUS_DEGREE)


class RestrictScalarsRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint


class LinearMapRankRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint


class ProjectiveLineRequest(StrictModel):
    presentation: FiniteFieldPresentation
    axis: Axis


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


class PaleyTournamentRequest(StrictModel):
    presentation: FiniteFieldPresentation


__all__ = [
    "CollisionRequest",
    "DirectionRankLedgerRequest",
    "FiberPartitionRequest",
    "FiniteMapTableRequest",
    "HomogeneousFixedSubspaceRequest",
    "LinearMapRankRequest",
    "OrbitDistributionRequest",
    "PaleyTournamentRequest",
    "PermutationRequest",
    "ProjectiveLineRequest",
    "RestrictScalarsRequest",
]
