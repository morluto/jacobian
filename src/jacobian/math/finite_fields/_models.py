"""Typed requests for atomic finite-field operations."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.finite_fields import (
    Axis,
    DirectionRankLedger,
    FiniteDimensionalSubspace,
    FiniteFieldPresentation,
    FiniteMapTable,
    FinitePolynomialMap,
    ProjectiveLine,
    ProjectivePoint,
)
from jacobian.math.finite_fields.values import _direction_rank_work


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


_MAX_PROJECTIVE_POINTS = 4_096
_MAX_FINITE_MAP_WORK = 1_000_000
_MAX_DIRECTION_RANK_WORK = 1_000_000


class RestrictScalarsRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint

    @model_validator(mode="after")
    def require_same_parent_and_axis(self) -> Self:
        if self.direction.presentation != self.subspace.presentation:
            raise _validation_error(
                "finite_field.direction_subspace_share_presentation",
                "direction and subspace must share their presentation",
            )
        if self.direction.axis != self.subspace.row_axis:
            raise _validation_error(
                "finite_field.direction_axis_match_subspace_row_axis",
                "direction axis must match the subspace row axis",
            )
        if _direction_rank_work(self.subspace, 1) > _MAX_DIRECTION_RANK_WORK:
            raise _validation_error(
                "finite_field.restriction_exceeds_operation_work_budget",
                "restriction exceeds the operation work budget",
            )
        return self


class LinearMapRankRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    direction: ProjectivePoint

    @model_validator(mode="after")
    def require_bound_direction(self) -> Self:
        if self.direction.presentation != self.subspace.presentation:
            raise _validation_error(
                "finite_field.direction_subspace_share_presentation",
                "direction and subspace must share their presentation",
            )
        if self.direction.axis != self.subspace.row_axis:
            raise _validation_error(
                "finite_field.direction_axis_match_subspace_row_axis",
                "direction axis must match the subspace row axis",
            )
        if _direction_rank_work(self.subspace, 1) > _MAX_DIRECTION_RANK_WORK:
            raise _validation_error(
                "finite_field.rank_derivation_exceeds_operation_work_budget",
                "rank derivation exceeds the operation work budget",
            )
        return self


class ProjectiveLineRequest(StrictModel):
    presentation: FiniteFieldPresentation
    axis: Axis

    @model_validator(mode="after")
    def require_two_coordinate_axis(self) -> Self:
        if len(self.axis.labels) != 2:
            raise _validation_error(
                "finite_field.projective_line_two_coordinate_axis",
                "projective-line enumeration requires a two-coordinate axis",
            )
        count = self.presentation.order + 1
        if count > _MAX_PROJECTIVE_POINTS:
            raise _validation_error(
                "finite_field.projective_line_exceeds_output_size_budget",
                "projective line exceeds the output-size budget",
            )
        return self


class DirectionRankLedgerRequest(StrictModel):
    subspace: FiniteDimensionalSubspace
    directions: ProjectiveLine

    @model_validator(mode="after")
    def require_same_parent_axis_and_work(self) -> Self:
        if self.directions.presentation != self.subspace.presentation:
            raise _validation_error(
                "finite_field.directions_subspace_share_presentation",
                "directions and subspace must share their presentation",
            )
        if self.directions.axis != self.subspace.row_axis:
            raise _validation_error(
                "finite_field.direction_axis_match_subspace_row_axis",
                "direction axis must match the subspace row axis",
            )
        if (
            _direction_rank_work(self.subspace, len(self.directions.points))
            > _MAX_DIRECTION_RANK_WORK
        ):
            raise _validation_error(
                "finite_field.direction_rank_ledger_exceeds_operation_work_budget",
                "direction-rank ledger exceeds the operation work budget",
            )
        return self


class OrbitDistributionRequest(StrictModel):
    ledger: DirectionRankLedger


class FiniteMapTableRequest(StrictModel):
    polynomial_map: FinitePolynomialMap

    @model_validator(mode="after")
    def require_bounded_enumeration(self) -> Self:
        work = (
            self.polynomial_map.domain.order
            * len(self.polynomial_map.polynomial.coefficients)
            * self.polynomial_map.domain.degree
        )
        if work > _MAX_FINITE_MAP_WORK:
            raise _validation_error(
                "finite_field.finite_map_exceeds_operation_work_budget",
                "finite map exceeds the operation work budget",
            )
        return self


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
