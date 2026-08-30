"""Public declaration for the exact bounded friable-enumerate operation."""

from pydantic_core import PydanticCustomError

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._friable_enumerate_kernels import (
    enumerate_friable as enumerate_friable,
)
from jacobian.math.number_theory._friable_enumerate_models import (
    FriableEnumerateRequest,
    FriableEnumerateResult,
)
from jacobian.math.number_theory._support import number_theory_operation


def compute_friable_enumerate(request: FriableEnumerateRequest) -> FriableEnumerateResult:
    try:
        family = enumerate_friable(
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

__all__ = ["FRIABLE_ENUMERATE_OPERATION"]
