"""Typed requests for atomic finite-field operations."""

from typing import Any

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel, canonicalize_json_containers
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


class _FiniteFieldRequest(StrictModel):
    """Strict wire request whose JSON arrays become canonical tuples."""

    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: Any) -> Any:
        return canonicalize_json_containers(data)


class HomogeneousFixedSubspaceRequest(_FiniteFieldRequest):
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
                    and all(isinstance(row, (list, tuple)) for row in matrix["entries"])
                )
                else matrix
                for matrix in matrices
            )
        return {**data, "action": normalized_action}


class RestrictScalarsRequest(_FiniteFieldRequest):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint


class LinearMapRankRequest(_FiniteFieldRequest):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint


class ProjectiveLineRequest(_FiniteFieldRequest):
    presentation: FiniteFieldPresentation
    axis: Axis


class DirectionRankLedgerRequest(_FiniteFieldRequest):
    subspace: FiniteDimensionalSubspace
    directions: ProjectiveLine


class OrbitDistributionRequest(_FiniteFieldRequest):
    ledger: DirectionRankLedger


class FiniteMapTableRequest(_FiniteFieldRequest):
    polynomial_map: FinitePolynomialMap


class FiberPartitionRequest(_FiniteFieldRequest):
    table: FiniteMapTable


class CollisionRequest(_FiniteFieldRequest):
    table: FiniteMapTable


class PermutationRequest(_FiniteFieldRequest):
    table: FiniteMapTable


class PaleyTournamentRequest(_FiniteFieldRequest):
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
