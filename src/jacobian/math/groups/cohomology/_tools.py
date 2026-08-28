"""Typed declarations for group cohomology operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.cohomology._models import (
    GroupCohomologyRequest,
    GroupCohomologyResult,
)
from jacobian.math.groups.cohomology._operations import compute_group_cohomology


def group_cohomology_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_COHOMOLOGY_EXAMPLE: dict[str, Any] = {
    "group": {
        "degree": 2,
        "generators": [[1, 0]],
    },
    "prime": 2,
    "max_degree": 2,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    group_cohomology_operation(
        "group_cohomology.cohomology.compute",
        "Compute group cohomology with trivial coefficients over GF(p)",
        "Given a finite permutation group G and a prime p, compute the "
        "cohomology groups H^n(G, K) with trivial coefficients K = GF(p) "
        "using the unnormalized inhomogeneous bar complex. Each group "
        "reports betti = dim H^n(G, K) exactly; cochain_dimension is the "
        "ambient cochain space dimension |G|^n of the unnormalized "
        "cochains, not the cohomology dimension. H^0 = K always; higher "
        "groups measure extensions, crossed homomorphisms, and "
        "obstruction classes.",
        GroupCohomologyRequest,
        GroupCohomologyResult,
        compute_group_cohomology,
        "group-cohomology",
        "cohomology",
        "exact",
        examples=(
            example(
                "z2_over_gf2",
                "Cohomology of Z/2 over GF(2); order^k stays inside the "
                "4096-element cochain and 65536-cell matrix budgets.",
                _COHOMOLOGY_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
