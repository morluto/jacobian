"""Wire contracts for finite-field algebraic sets (#3726)."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.finite_fields._algebraic_sets import (
    AffinePoint,
    FieldEmbedding,
    PolynomialSystem,
)
from jacobian.math.finite_fields.values import ProjectivePoint


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class _AlgebraicSetRequest(StrictModel):
    @model_validator(mode="before")
    @classmethod
    def normalize_json_containers(cls, data: Any) -> Any:
        return canonicalize_json_containers(data)


class AffineZeroSetRequest(_AlgebraicSetRequest):
    """Enumerate all simultaneous zeros of a supplied affine system."""

    system: PolynomialSystem


class AffineZeroSetResult(_AlgebraicSetRequest):
    """Complete affine zero locus with compact agreeing count."""

    system: PolynomialSystem
    points: tuple[AffinePoint, ...] = Field(max_length=65_536)
    point_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        if self.point_count != len(self.points):
            raise _error(
                "finite_field.algebraic_set_count_mismatch",
                "point_count must equal the retained point family",
            )
        for point in self.points:
            if point.presentation != self.system.presentation:
                raise _error(
                    "finite_field.algebraic_set_point_parent",
                    "zero-set points must share the system presentation",
                )
            if point.variable_axis != self.system.variable_axis:
                raise _error(
                    "finite_field.algebraic_set_point_axis",
                    "zero-set points must share the system variable axis",
                )
        if len({tuple(p.coordinates) for p in self.points}) != len(self.points):
            raise _error(
                "finite_field.algebraic_set_duplicate_point",
                "zero-set points must be distinct",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, system: PolynomialSystem, points: tuple[AffinePoint, ...]
    ) -> Self:
        return cls.model_construct(
            system=system, points=points, point_count=len(points)
        )


class AffineZeroCountRequest(_AlgebraicSetRequest):
    system: PolynomialSystem


class AffineZeroCountResult(_AlgebraicSetRequest):
    system: PolynomialSystem
    point_count: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, *, system: PolynomialSystem, point_count: int) -> Self:
        return cls.model_construct(system=system, point_count=point_count)


class ProjectiveZeroSetRequest(_AlgebraicSetRequest):
    """Enumerate canonical scalar classes of a homogeneous system."""

    system: PolynomialSystem


class ProjectiveZeroSetResult(_AlgebraicSetRequest):
    system: PolynomialSystem
    points: tuple[ProjectivePoint, ...] = Field(max_length=65_536)
    point_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        if self.point_count != len(self.points):
            raise _error(
                "finite_field.algebraic_set_count_mismatch",
                "point_count must equal the retained point family",
            )
        for point in self.points:
            if point.presentation != self.system.presentation:
                raise _error(
                    "finite_field.algebraic_set_point_parent",
                    "projective points must share the system presentation",
                )
        if len({point.digest for point in self.points}) != len(self.points):
            raise _error(
                "finite_field.algebraic_set_duplicate_point",
                "projective points must be distinct scalar classes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, system: PolynomialSystem, points: tuple[ProjectivePoint, ...]
    ) -> Self:
        return cls.model_construct(
            system=system, points=points, point_count=len(points)
        )


class ProjectiveZeroCountRequest(_AlgebraicSetRequest):
    system: PolynomialSystem


class ProjectiveZeroCountResult(_AlgebraicSetRequest):
    system: PolynomialSystem
    point_count: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, *, system: PolynomialSystem, point_count: int) -> Self:
        return cls.model_construct(system=system, point_count=point_count)


class BaseChangeRequest(_AlgebraicSetRequest):
    """Transport a system along an explicit exact field embedding."""

    system: PolynomialSystem
    embedding: FieldEmbedding


class BaseChangeResult(_AlgebraicSetRequest):
    system: PolynomialSystem
    embedding: FieldEmbedding
    transported: PolynomialSystem

    @model_validator(mode="after")
    def require_transport_binding(self) -> Self:
        if self.system.presentation != self.embedding.source:
            raise _error(
                "finite_field.base_change_source_mismatch",
                "system presentation must equal the embedding source",
            )
        if self.transported.presentation != self.embedding.target:
            raise _error(
                "finite_field.base_change_target_mismatch",
                "transported system must use the embedding target",
            )
        if self.transported.variable_axis != self.system.variable_axis:
            raise _error(
                "finite_field.base_change_axis_mismatch",
                "transported system must preserve the variable axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        system: PolynomialSystem,
        embedding: FieldEmbedding,
        transported: PolynomialSystem,
    ) -> Self:
        return cls.model_construct(
            system=system, embedding=embedding, transported=transported
        )


__all__ = [
    "AffineZeroCountRequest",
    "AffineZeroCountResult",
    "AffineZeroSetRequest",
    "AffineZeroSetResult",
    "BaseChangeRequest",
    "BaseChangeResult",
    "ProjectiveZeroCountRequest",
    "ProjectiveZeroCountResult",
    "ProjectiveZeroSetRequest",
    "ProjectiveZeroSetResult",
]
