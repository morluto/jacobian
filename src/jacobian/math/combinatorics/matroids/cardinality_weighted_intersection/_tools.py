"""Operation declaration for lexicographic matroid intersection."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.matroids._models import (
    MatroidWeightedIntersectionOptimizationRequest,
)

from ._models import MatroidCardinalityWeightedIntersectionResult
from .operations import maximum_cardinality_weighted_matroid_intersection

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matroid.intersection.maximum_cardinality_weighted.compute",
        title="Optimize weight among maximum-cardinality common independent sets",
        description=(
            "Return a common independent set of maximum possible cardinality, "
            "then maximum original integer weight among sets of that size. "
            "The exact result retains the shifted weighted-intersection witness "
            "and the bonus needed to recover the original objective."
        ),
        request_type=MatroidWeightedIntersectionOptimizationRequest,
        result_type=MatroidCardinalityWeightedIntersectionResult,
        run=maximum_cardinality_weighted_matroid_intersection,
        tags=(
            "matroid",
            "intersection",
            "maximum-cardinality",
            "maximum-weight",
            "exact",
        ),
        discovery_terms=(
            "maximum weight maximum cardinality common independent set",
            "maximum cardinality weighted matroid intersection",
            "fixed cardinality maximum weight matroid intersection",
            "lexicographic cardinality weight optimization",
        ),
        examples=(
            OperationExample(
                name="negative_weights_do_not_shrink_the_common_set",
                description=(
                    "Cardinality is optimized before weight, so a negative "
                    "weight is retained when it permits a larger common set."
                ),
                input={
                    "first": {
                        "matrix": {"prime": 2, "entries": [[1, 0, 0]], "columns": 3},
                        "ground_labels": ["a", "b", "c"],
                    },
                    "second": {
                        "matrix": {"prime": 2, "entries": [[0, 1, 1]], "columns": 3},
                        "ground_labels": ["a", "b", "c"],
                    },
                    "weight_function": {
                        "ground_axis": ["a", "b", "c"],
                        "values": [-5, 1, 0],
                    },
                },
            ),
        ),
    ),
)
