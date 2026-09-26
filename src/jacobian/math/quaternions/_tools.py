"""Public exact rational unit-quaternion operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.quaternions._models import (
    RationalUnitQuaternion,
    RationalUnitQuaternionBinaryRequest,
    RationalUnitQuaternionUnaryRequest,
)
from jacobian.math.quaternions.operations import (
    conjugate_rational_unit_quaternion,
    multiply_rational_unit_quaternions,
)


def _multiply(request: RationalUnitQuaternionBinaryRequest) -> RationalUnitQuaternion:
    return multiply_rational_unit_quaternions(request.left, request.right)


def _conjugate(request: RationalUnitQuaternionUnaryRequest) -> RationalUnitQuaternion:
    return conjugate_rational_unit_quaternion(request.value)


_I = {
    "coordinates": [
        {"num": "0", "den": "1"},
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
    ]
}
_J = {
    "coordinates": [
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
    ]
}
_Q = {
    "coordinates": [
        {"num": "3", "den": "5"},
        {"num": "4", "den": "5"},
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
    ]
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quaternion.rational_unit.multiply.compute",
        title="Multiply rational unit quaternions",
        description="Compute the exact Hamilton product of two norm-one quaternions over QQ.",
        request_type=RationalUnitQuaternionBinaryRequest,
        result_type=RationalUnitQuaternion,
        run=_multiply,
        tags=("quaternion", "group", "exact"),
        discovery_terms=("rational unit quaternion product", "Hamilton multiplication"),
        examples=(
            OperationExample(
                name="i_times_j",
                description="The Hamilton product i*j is k.",
                input={"left": _I, "right": _J},
            ),
        ),
    ),
    MathTool(
        operation_id="quaternion.rational_unit.conjugate.compute",
        title="Conjugate a rational unit quaternion",
        description=(
            "Return the exact quaternion conjugate (a,-b,-c,-d), which is the "
            "group inverse of a norm-one rational quaternion."
        ),
        request_type=RationalUnitQuaternionUnaryRequest,
        result_type=RationalUnitQuaternion,
        run=_conjugate,
        tags=("quaternion", "group", "exact"),
        discovery_terms=(
            "rational unit quaternion conjugate",
            "rational unit quaternion group inverse",
        ),
        examples=(
            OperationExample(
                name="conjugate_three_fifths",
                description="Conjugate, equivalently invert, (3/5,4/5,0,0).",
                input={"value": _Q},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
