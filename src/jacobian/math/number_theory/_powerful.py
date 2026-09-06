"""Exact bounded powerful-number decision and declaration."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._powerful_kernels import decide_powerful_data
from jacobian.math.number_theory._powerful_models import (
    PowerfulNumberRequest,
    PowerfulNumberResult,
)


def decide_powerful(request: PowerfulNumberRequest) -> PowerfulNumberResult:
    """Return the exact source-bound powerful-number decision."""

    data = decide_powerful_data(request.value)
    return PowerfulNumberResult._from_kernel(request, data=data)


POWERFUL_NUMBER_OPERATION = MathTool(
    operation_id="integer.decide.powerful",
    title="Decide powerful-number status",
    description="Decide exactly whether every prime exponent of one positive integer is at "
    "least two. Trial division through the derived cutoff B=ceil(n^(1/5)) and "
    "exact perfect-power classification of the B-rough residual avoid complete "
    "factorization while returning a source-bound partial certificate.",
    request_type=PowerfulNumberRequest,
    result_type=PowerfulNumberResult,
    run=decide_powerful,
    tags=(
        "number-theory",
        "2-full",
        "powerful",
        "powerful-number",
        "predicate",
        "certificate",
    ),
    examples=(
        OperationExample(
            name="powerful_12168",
            description="Decide whether 12168 is powerful from a bounded partial-factor "
            "certificate; the value must be a positive canonical integer with "
            "at most 25 digits.",
            input={"value": "12168"},
        ),
    ),
)
