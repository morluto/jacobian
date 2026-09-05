"""Triangle-free diameter augmentation public API."""

from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationBudget,
    TriangleFreeDiameterAugmentationResult,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation.operations import (
    triangle_free_diameter_augmentation,
)

__all__ = [
    "TriangleFreeDiameterAugmentationBudget",
    "TriangleFreeDiameterAugmentationResult",
    "triangle_free_diameter_augmentation",
]
