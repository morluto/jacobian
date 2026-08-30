"""Typed requests for atomic finite-field operations."""

from typing import Any

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.finite_fields.values import (
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
    degree: StrictInt = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_json_array_fields(cls, data: Any) -> Any:
        """Convert transport arrays to the immutable tuple fields before strict parsing."""

        if not isinstance(data, dict) or not isinstance(data.get("action"), dict):
            return data
        action = data["action"]
        normalized_action = dict(action)
        axis = action.get("variable_axis")
        if isinstance(axis, dict) and isinstance(axis.get("labels"), list):
            normalized_action["variable_axis"] = {
                **axis,
                "labels": tuple(axis["labels"]),
            }
        matrices = action.get("generator_matrices")
        if isinstance(matrices, list):
            normalized_action["generator_matrices"] = tuple(
                {
                    **matrix,
                    "entries": tuple(tuple(row) for row in matrix["entries"]),
                }
                if (
                    isinstance(matrix, dict)
                    and isinstance(matrix.get("entries"), list)
                    and all(
                        isinstance(row, (list, tuple))
                        for row in matrix["entries"]
                    )
                )
                else matrix
                for matrix in matrices
            )
        return {**data, "action": normalized_action}


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
