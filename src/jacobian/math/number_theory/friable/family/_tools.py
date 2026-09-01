"""Public declaration for the exact bounded friable-family operation."""

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.friable.family._models import (
    FriableFamilyRequest,
    FriableFamilyResult,
)
from jacobian.math.number_theory.friable.family.operations import (
    enumerate_friable_family_kernel,
)


def compute_friable_family(request: FriableFamilyRequest) -> FriableFamilyResult:
    try:
        family = enumerate_friable_family_kernel(
            parse_canonical_integer(request.x),
            parse_canonical_integer(request.y),
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("x", "y"),
            code="number_theory.friable_family_admission_error",
            message=str(exc),
        ) from exc
    return FriableFamilyResult._from_kernel(request, family=family)


TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.friable.family.enumerate",
        title="Enumerate friable integers exactly",
        description=(
            "Return the complete increasing tuple of positive y-friable "
            "integers at most x, i.e. integers whose prime factors are all "
            "at most the inclusive cutoff y. The result retains x and y "
            "alongside the exact ordered family from the admitted materialized "
            "or generated-search envelope."
        ),
        request_type=FriableFamilyRequest,
        result_type=FriableFamilyResult,
        run=compute_friable_family,
        tags=(
            "number-theory",
            "friable",
            "smooth-number",
            "enumeration",
            "exact",
        ),
        examples=(
            OperationExample(
                name="five_friable_through_100",
                description=(
                    "Enumerate the positive 5-friable integers at most 100; x "
                    "and y must be canonical nonnegative decimals and the "
                    "selected exact enumeration regime must fit its work "
                    "budget."
                ),
                input={"x": "100", "y": "5"},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
