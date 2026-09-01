"""Typed declarations for the periodic congruence interval count operation."""

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.periodic_interval_count._models import (
    PeriodicIntervalCountRequest,
    PeriodicIntervalCountResult,
)
from jacobian.math.number_theory.periodic_interval_count.operations import (
    compute_periodic_interval_count,
)


def _compute(request: PeriodicIntervalCountRequest) -> PeriodicIntervalCountResult:
    if max(len(request.lower.lstrip("-")), len(request.upper.lstrip("-"))) > 2_000_000:
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="number_theory.periodic.endpoint_work_bound",
            message="periodic interval endpoints exceed the digit-sensitive parsing work bound",
        )
    return compute_periodic_interval_count(
        request.source,
        parse_canonical_integer(request.lower),
        parse_canonical_integer(request.upper),
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="congruence.periodic_union.interval_count.compute",
        title="Count union or complement members in a closed integer interval",
        description=(
            "Given a normalized finite periodic congruence union/complement and "
            "a closed integer interval [lower, upper], return the exact number "
            "of integers in that interval belonging to the declared periodic set."
        ),
        request_type=PeriodicIntervalCountRequest,
        result_type=PeriodicIntervalCountResult,
        run=_compute,
        tags=("number-theory", "congruence", "exact"),
        examples=(
            OperationExample(
                name="simple",
                description="Count multiples of 3 in [1, 20].",
                input={
                    "source": {
                        "subsets": [{"modulus": "3", "residues": ["0"]}],
                        "complement": False,
                    },
                    "lower": "1",
                    "upper": "20",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
