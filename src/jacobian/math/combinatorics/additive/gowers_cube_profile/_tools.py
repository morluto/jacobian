"""Typed declarations for the Gowers cube profile operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.additive.gowers_cube_profile._models import (
    GowersCubeRequest,
    GowersCubeResult,
)
from jacobian.math.combinatorics.additive.gowers_cube_profile.operations import (
    compute_gowers_cube_profile,
)


def _compute(request: GowersCubeRequest) -> GowersCubeResult:
    return compute_gowers_cube_profile(request.modulus, request.subset, request.order)


TOOLS: MathTools = (
    MathTool(
        operation_id="additive.finite_abelian_subset.gowers_cube_profile.compute",
        title="Compute exact finite-Abelian Gowers cube counts",
        description=(
            "For one supplied subset A of Z/mZ and one admitted order s, "
            "return the exact count of all labelled affine s-cubes whose "
            "every vertex lies in A."
        ),
        request_type=GowersCubeRequest,
        result_type=GowersCubeResult,
        run=_compute,
        tags=("additive", "gowers", "cube", "exact"),
        examples=(
            example(
                "z5_full",
                "Gowers U^2 cubes for full subset of Z/5Z.",
                {"modulus": 5, "subset": [0, 1, 2, 3, 4], "order": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
