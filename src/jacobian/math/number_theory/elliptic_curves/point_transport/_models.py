"""Typed request and result for finite-field point transport."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory.elliptic_curves.finite_field import (
    FiniteFieldEllipticPoint,
    FiniteFieldIsomorphismResult,
)


class FiniteFieldPointTransportRequest(StrictModel):
    """Apply a supplied model-isomorphism witness to one source point."""

    isomorphism: FiniteFieldIsomorphismResult
    point: FiniteFieldEllipticPoint

    @model_validator(mode="after")
    def source_point_is_bound(self) -> Self:
        if self.point.curve != self.isomorphism.source:
            raise ValueError("point must belong to the isomorphism source curve")
        return self


class FiniteFieldPointTransportResult(StrictModel):
    """The exact image point, retaining its map and source point."""

    isomorphism: FiniteFieldIsomorphismResult
    source_point: FiniteFieldEllipticPoint
    target_point: FiniteFieldEllipticPoint

    @model_validator(mode="after")
    def points_retain_curve_parents(self) -> Self:
        if (
            not self.isomorphism.isomorphic
            or self.isomorphism.scaling is None
            or self.source_point.curve != self.isomorphism.source
            or self.target_point.curve != self.isomorphism.target
            or self.source_point.at_infinity != self.target_point.at_infinity
        ):
            raise ValueError("transported points must retain the isomorphism endpoints")
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        isomorphism: FiniteFieldIsomorphismResult,
        source_point: FiniteFieldEllipticPoint,
        target_point: FiniteFieldEllipticPoint,
    ) -> Self:
        """Construct after the admitted kernel establishes the image relation."""

        return cls.model_construct(
            isomorphism=isomorphism,
            source_point=source_point,
            target_point=target_point,
        )


__all__ = ["FiniteFieldPointTransportRequest", "FiniteFieldPointTransportResult"]
