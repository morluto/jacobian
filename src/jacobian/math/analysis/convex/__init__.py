"""Convex-analysis operation ownership."""

from jacobian.math.analysis.convex.operations import (
    max_affine_evaluation,
    max_affine_subdifferential,
)

__all__: list[str] = ["max_affine_evaluation", "max_affine_subdifferential"]
