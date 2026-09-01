"""Typed declarations for group cohomology operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.cohomology._models import (
    GroupCohomologyRequest,
    GroupCohomologyResult,
)
from jacobian.math.groups.cohomology.operations import group_cohomology


def _run_group_cohomology(
    request: GroupCohomologyRequest,
) -> GroupCohomologyResult:
    return group_cohomology(request.group, request.prime, request.max_degree)


_COHOMOLOGY_EXAMPLE: dict[str, Any] = {
    "group": {
        "degree": 2,
        "generators": [[1, 0]],
    },
    "prime": 2,
    "max_degree": 2,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="group_cohomology.cohomology.compute",
        title="Compute group cohomology with trivial coefficients over GF(p)",
        description="Given a finite permutation group G and a prime p, compute the "
        "cohomology groups H^n(G, K) with trivial coefficients K = GF(p) "
        "using the unnormalized inhomogeneous bar complex. Each group "
        "reports betti = dim H^n(G, K) exactly; cochain_dimension is the "
        "ambient cochain space dimension |G|^n of the unnormalized "
        "cochains, not the cohomology dimension. H^0 = K always; higher "
        "groups measure extensions, crossed homomorphisms, and "
        "obstruction classes.",
        request_type=GroupCohomologyRequest,
        result_type=GroupCohomologyResult,
        run=_run_group_cohomology,
        tags=("group-cohomology", "cohomology", "exact"),
        examples=(
            OperationExample(
                name="z2_over_gf2",
                description="Cohomology of Z/2 over GF(2); order^k stays inside the "
                "4096-element cochain and 65536-cell matrix budgets.",
                input=_COHOMOLOGY_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
