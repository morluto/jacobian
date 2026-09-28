"""Catalog declaration for filtered direct sums."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.chain_complexes.filtered_direct_sum import (
    FilteredDirectSumRequest,
    FilteredDirectSumResult,
    filtered_direct_sum,
)


def _run(request: FilteredDirectSumRequest) -> FilteredDirectSumResult:
    return filtered_direct_sum(request.left, request.right)


_ONE_STEP = {
    "complex": {
        "coefficient_ring": "QQ",
        "degree_min": 0,
        "degree_max": 0,
        "basis_sizes": [1],
        "differential_matrices": [],
    },
    "filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
}


TOOLS = (
    MathTool(
        operation_id="homological.filtered_chain_complex.direct_sum.compute",
        title="Form the direct sum of filtered chain complexes",
        description=(
            "Form the componentwise filtered direct sum of two finite based "
            "filtered chain complexes over the same exact field, degree interval, "
            "and filtration axis. Return the block-sum differential, the direct "
            "sum filtration, and both canonical chain inclusions."
        ),
        request_type=FilteredDirectSumRequest,
        result_type=FilteredDirectSumResult,
        run=_run,
        tags=("homological", "filtered-complex", "direct-sum", "exact"),
        discovery_terms=(
            "filtered complex direct sum",
            "direct sum filtration",
            "sum of filtered chain complexes",
        ),
        examples=(
            OperationExample(
                name="sum_of_one_dimensional_filtered_complexes",
                description=(
                    "Take the direct sum of two one-dimensional zero-differential "
                    "filtered complexes over QQ."
                ),
                input={"left": _ONE_STEP, "right": _ONE_STEP},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
