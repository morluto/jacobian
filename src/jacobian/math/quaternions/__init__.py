"""Exact rational unit-quaternion values and operations."""

from jacobian.math.quaternions._models import RationalUnitQuaternion
from jacobian.math.quaternions.operations import (
    conjugate_rational_unit_quaternion,
    inverse_rational_unit_quaternion,
    multiply_rational_unit_quaternions,
    rational_unit_quaternion_scalar_part,
)

__all__ = [
    "RationalUnitQuaternion",
    "conjugate_rational_unit_quaternion",
    "inverse_rational_unit_quaternion",
    "multiply_rational_unit_quaternions",
    "rational_unit_quaternion_scalar_part",
]
