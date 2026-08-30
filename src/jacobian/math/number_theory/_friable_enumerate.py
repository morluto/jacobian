"""Public declaration for the exact bounded friable-enumerate operation."""

from pydantic_core import PydanticCustomError

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._friable_enumerate_kernels import (
    enumerate_friable as _enumerate_friable_kernel,
)
from jacobian.math.number_theory._friable_enumerate_models import (
    FriableEnumerateRequest,
    FriableEnumerateResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def compute_friable_enumerate(
    request: FriableEnumerateRequest,
) -> FriableEnumerateResult:
    try:
        family = _enumerate_friable_kernel(
            parse_canonical_integer(request.x),
            parse_canonical_integer(request.y),
        )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("x", "y"),
            code=exc.type,
            message=exc.message(),
        ) from exc
    return FriableEnumerateResult._from_kernel(request, family=family)


def enumerate_friable(
    x: int | IntegerValue, y: int | IntegerValue
) -> FriableEnumerateResult:
    """Enumerate friable integers from native integer inputs."""

    x_value = parse_canonical_integer(x.value) if isinstance(x, IntegerValue) else x
    y_value = parse_canonical_integer(y.value) if isinstance(y, IntegerValue) else y
    request = FriableEnumerateRequest(
        x=format_canonical_integer(x_value), y=format_canonical_integer(y_value)
    )
    return compute_friable_enumerate(request)


FRIABLE_ENUMERATE_OPERATION = number_theory_operation(
    "integer.friable.enumerate",
    "Enumerate friable integers exactly",
    (
        "Return the complete increasing tuple of positive integers at most x "
        "whose prime factors are all at most the inclusive cutoff y. The kernel "
        "generates prime-exponent vectors whose product is at most x, maps them "
        "to positive integer products, and sorts/deduplicates canonically."
    ),
    FriableEnumerateRequest,
    FriableEnumerateResult,
    compute_friable_enumerate,
    "number-theory",
    "friable",
    "smooth-number",
    "enumeration",
    "exact",
    examples=(
        example(
            "five_friable_through_20",
            (
                "Enumerate the positive 5-friable integers at most 20; x and y "
                "must be canonical nonnegative decimals and the selected exact "
                "enumeration regime must fit its work budget."
            ),
            {"x": "20", "y": "5"},
        ),
    ),
)

__all__ = ["FRIABLE_ENUMERATE_OPERATION", "enumerate_friable"]
