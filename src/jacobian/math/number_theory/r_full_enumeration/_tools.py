"""Typed declarations for the r-full enumeration operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.number_theory.r_full_enumeration._models import (
    RFullEnumerationRequest,
    RFullEnumerationResult,
)
from jacobian.math.number_theory.r_full_enumeration.operations import (
    enumerate_r_full,
)


def _enumerate(request: RFullEnumerationRequest) -> RFullEnumerationResult:
    return enumerate_r_full(request.bound, request.minimum_exponent)


TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.r_full.enumerate",
        title="Enumerate bounded r-full integer families",
        description=(
            "For a positive bound and minimum exponent r >= 2, return the "
            "complete sorted tuple of positive r-full integers at most the "
            "bound — integers whose every prime divisor occurs to exponent "
            "at least r."
        ),
        request_type=RFullEnumerationRequest,
        result_type=RFullEnumerationResult,
        run=_enumerate,
        tags=("number_theory", "r_full", "powerful", "exact"),
        examples=(
            example(
                "powerful_100",
                "Powerful (2-full) integers up to 100.",
                {"bound": 100, "minimum_exponent": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
