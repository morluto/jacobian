"""Public declaration for finite simplicial-set skeleta."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.skeleton import (
    SimplicialSetSkeletonRequest,
    SimplicialSetSkeletonResult,
    simplicial_set_skeleton,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

_DELTA_TWO = standard_simplex(2, 2).model_dump(mode="json")


def _run(request: SimplicialSetSkeletonRequest) -> SimplicialSetSkeletonResult:
    return simplicial_set_skeleton(request.simplicial_set, request.k)


TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.skeleton.compute",
        title="Construct a finite simplicial-set skeleton",
        description=(
            "Return the k-skeleton of a finite truncated simplicial set through "
            "its existing maximum degree, together with its exact inclusion "
            "map. The skeleton is generated from all simplices in degrees at "
            "most k by the visible degeneracy maps; no claim is made above the "
            "source prefix."
        ),
        request_type=SimplicialSetSkeletonRequest,
        result_type=SimplicialSetSkeletonResult,
        run=_run,
        tags=("topology", "simplicial-set", "skeleton", "exact"),
        discovery_terms=("simplicial set skeleton", "k-skeleton", "degenerate simplex"),
        examples=(
            OperationExample(
                name="one_skeleton_of_delta_two",
                description=(
                    "Construct the 1-skeleton of Delta[2] through degree 2; "
                    "degree 2 contains exactly the degeneracies of vertices "
                    "and edges."
                ),
                input={"simplicial_set": _DELTA_TWO, "k": 1},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
