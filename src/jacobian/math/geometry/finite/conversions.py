"""Explicit maps from finite-geometry values into finite-field values."""

from __future__ import annotations

from jacobian.math.finite_fields import (
    Axis,
    FiniteFieldElement,
    FiniteFieldPresentation,
)
from jacobian.math.finite_fields import (
    ProjectivePoint as FiniteFieldProjectivePoint,
)
from jacobian.math.geometry.finite.values import ProjectivePoint


def embed_projective_point_in_finite_field(
    point: ProjectivePoint,
    presentation: FiniteFieldPresentation,
    axis: Axis,
) -> FiniteFieldProjectivePoint:
    """Embed a canonical point of ``PG(n, p)`` into a chosen extension field.

    The target field and axis are explicit because this map changes the scalar
    parent from ``F_p`` to the caller's chosen ``F_{p^d}``.  Coordinate labels
    retain their source order, while the target ``Axis`` supplies the semantic
    name required by finite-field linear-algebra values.
    """

    if presentation.characteristic != point.space.field_order:
        raise ValueError(
            "finite-field presentation characteristic must match the prime-field point"
        )
    if axis.labels != point.space.axis:
        raise ValueError(
            "finite-field axis labels must match the prime-field point axis"
        )

    zero_tail = (0,) * (presentation.degree - 1)
    return FiniteFieldProjectivePoint(
        presentation=presentation,
        axis=axis,
        coordinates=tuple(
            FiniteFieldElement(
                presentation=presentation,
                coordinates=(coordinate, *zero_tail),
            )
            for coordinate in point.coordinates
        ),
    )


__all__ = ["embed_projective_point_in_finite_field"]
