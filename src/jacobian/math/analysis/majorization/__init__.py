"""Majorization and matrix mixing operations."""

from jacobian.math.analysis.majorization._models import RationalVector
from jacobian.math.analysis.majorization.operations import (
    birkhoff_decomposition,
    doubly_stochastic_check,
    majorization_check,
    schur_horn_check,
    t_transform_sequence,
    verify_birkhoff,
    verify_doubly_stochastic,
    verify_majorization,
    verify_schur_horn,
    verify_weak_majorization,
    weak_majorization_check,
)

__all__ = [
    "RationalVector",
    "birkhoff_decomposition",
    "doubly_stochastic_check",
    "majorization_check",
    "schur_horn_check",
    "t_transform_sequence",
    "verify_birkhoff",
    "verify_doubly_stochastic",
    "verify_majorization",
    "verify_schur_horn",
    "verify_weak_majorization",
    "weak_majorization_check",
]
