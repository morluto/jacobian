"""Public declaration for the exact bounded friable-count operation."""

from pydantic_core import PydanticCustomError

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory._friable_kernel import count_friable as count_friable
from jacobian.math.number_theory._friable_models import (
    FriableCountRequest,
    FriableCountResult,
)


def compute_friable_count(request: FriableCountRequest) -> FriableCountResult:
    try:
        count = count_friable(
            parse_canonical_integer(request.x),
            parse_canonical_integer(request.y),
        )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("x", "y"),
            code=exc.type,
            message=exc.message(),
        ) from exc
    return FriableCountResult._from_kernel(request, count=count)


FRIABLE_COUNT_OPERATION = MathTool(
    operation_id="number_theory.friable.count.compute",
    title="Count friable integers exactly",
    description=(
        "Return the exact number Psi(x, y) of positive integers at most x whose "
        "prime factors are all at most the inclusive cutoff y. The result retains "
        "x and y alongside the exact count from the admitted materialized or "
        "generated-search envelope."
    ),
    request_type=FriableCountRequest,
    result_type=FriableCountResult,
    run=compute_friable_count,
    tags=("number-theory", "friable", "smooth-number", "counting", "exact"),
    examples=(
        OperationExample(
            name="five_friable_through_100",
            description=(
                "Count the positive 5-friable integers at most 100; x and y "
                "must be canonical nonnegative decimals and the selected exact "
                "counting regime must fit its work budget."
            ),
            input={"x": "100", "y": "5"},
        ),
    ),
)

__all__ = ["FRIABLE_COUNT_OPERATION"]
