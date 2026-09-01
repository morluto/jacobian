"""Public declaration for the exact bounded friable-enumerate operation."""

from pydantic_core import PydanticCustomError

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory._friable_enumerate_kernels import (
    enumerate_friable as _enumerate_friable_kernel,
)
from jacobian.math.number_theory._friable_enumerate_models import (
    FriableEnumerateRequest,
    FriableEnumerateResult,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def compute_friable_enumerate(
    request: FriableEnumerateRequest,
) -> FriableEnumerateResult:
    x_value = parse_canonical_integer(request.x)
    y_value = parse_canonical_integer(request.y)
    try:
        family = _enumerate_friable_kernel(x_value, y_value)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("x", "y"), code=exc.type, message=exc.message()
        ) from exc
    return FriableEnumerateResult._from_kernel_values(
        x_value,
        y_value,
        family=family,
    )


def enumerate_friable(
    x: int | IntegerValue, y: int | IntegerValue
) -> FriableEnumerateResult:
    """Enumerate friable integers from native integer inputs."""

    x_value = parse_canonical_integer(x.value) if isinstance(x, IntegerValue) else x
    y_value = parse_canonical_integer(y.value) if isinstance(y, IntegerValue) else y
    try:
        family = _enumerate_friable_kernel(x_value, y_value)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("x", "y"), code=exc.type, message=exc.message()
        ) from exc
    return FriableEnumerateResult._from_kernel_values(
        x_value,
        y_value,
        family=family,
    )


FRIABLE_ENUMERATE_OPERATION = MathTool(
    operation_id="integer.friable.enumerate",
    title="Enumerate friable integers exactly",
    description=(
        "Return the complete increasing tuple of positive integers at most x "
        "whose prime factors are all at most the inclusive cutoff y. The kernel "
        "generates prime-exponent vectors whose product is at most x, maps them "
        "to positive integer products, and sorts/deduplicates canonically."
    ),
    request_type=FriableEnumerateRequest,
    result_type=FriableEnumerateResult,
    run=compute_friable_enumerate,
    tags=("number-theory", "friable", "smooth-number", "enumeration", "exact"),
    examples=(
        OperationExample(
            name="five_friable_through_20",
            description=(
                "Enumerate the positive 5-friable integers at most 20; x and y "
                "must be canonical nonnegative decimals and the selected exact "
                "enumeration regime must fit its work budget."
            ),
            input={"x": "20", "y": "5"},
        ),
    ),
)

__all__ = ["FRIABLE_ENUMERATE_OPERATION", "enumerate_friable"]
