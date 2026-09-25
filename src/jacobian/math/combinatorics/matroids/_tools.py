"""Linear matroid operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.matroids._models import (
    GraphicMatroidRequest,
    LinearMatroid,
    MatroidClosureRequest,
    MatroidClosureResult,
    MatroidCommonBasisRequest,
    MatroidCommonBasisResult,
    MatroidIntersectionRequest,
    MatroidIntersectionResult,
    MatroidWeightedIntersectionCertificateRequest,
    MatroidWeightedIntersectionResult,
    MaximumWeightBasisRequest,
    MaximumWeightBasisResult,
    MaximumWeightIndependentSetRequest,
    MaximumWeightIndependentSetResult,
)
from jacobian.math.combinatorics.matroids.graphic import graphic_matroid
from jacobian.math.combinatorics.matroids.intersection import (
    matroid_common_basis,
    matroid_intersection,
    weighted_intersection_certificate,
)
from jacobian.math.combinatorics.matroids.operations import (
    closure_result,
    maximum_weight_basis_result,
    maximum_weight_independent_set_result,
)


def _run_closure(request: MatroidClosureRequest) -> MatroidClosureResult:
    return closure_result(request.matroid, request.subset)


def _run_maximum_weight_basis(
    request: MaximumWeightBasisRequest,
) -> MaximumWeightBasisResult:
    return maximum_weight_basis_result(request.matroid, request.weight_function)


def _run_maximum_weight_independent_set(
    request: MaximumWeightIndependentSetRequest,
) -> MaximumWeightIndependentSetResult:
    return maximum_weight_independent_set_result(
        request.matroid, request.weight_function
    )


def _run_common_basis(
    request: MatroidCommonBasisRequest,
) -> MatroidCommonBasisResult:
    return matroid_common_basis(request.first, request.second)


def _run_weighted_intersection_certificate(
    request: MatroidWeightedIntersectionCertificateRequest,
) -> MatroidWeightedIntersectionResult:
    return weighted_intersection_certificate(request)


_CLOSURE_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "matrix": {
            "prime": 5,
            "entries": [[1, 0, 1], [0, 1, 1]],
            "columns": 3,
        },
    },
    "subset": [0, 1],
}

_MAXIMUM_WEIGHT_BASIS_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "matrix": {
            "prime": 5,
            "entries": [[1, 0, 1], [0, 1, 1]],
            "columns": 3,
        },
    },
    "weight_function": {
        "ground_axis": ["0", "1", "2"],
        "values": [3, 2, 5],
    },
}

_MAXIMUM_WEIGHT_INDEPENDENT_SET_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "matrix": {
            "prime": 5,
            "entries": [[1, 0, 1], [0, 1, 1]],
            "columns": 3,
        },
    },
    "weight_function": {
        "ground_axis": ["0", "1", "2"],
        "values": [4, -3, 2],
    },
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matroid.graphic.represent.compute",
        title="Represent a simple graph as a binary linear matroid",
        description=(
            "Return the GF(2) vertex-edge incidence representation of a simple "
            "graph. Each column is one graph edge; the canonical ground labels "
            "are compact JSON endpoint pairs. Independent edge sets are forests, "
            "so bases of disconnected graphs are spanning forests. The operation "
            "admits at most 256 vertices and 256 edges before matrix expansion."
        ),
        request_type=GraphicMatroidRequest,
        result_type=LinearMatroid,
        run=graphic_matroid,
        tags=("matroid", "graphic-matroid", "graph", "GF(2)", "exact"),
        discovery_terms=(
            "graphic matroid of a graph",
            "graph edges as a binary linear matroid",
            "spanning forest matroid representation",
        ),
        examples=(
            OperationExample(
                name="triangle_graphic_matroid",
                description=(
                    "Represent the three edges of a triangle by their GF(2) "
                    "vertex-edge incidence columns."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matroid.closure.compute",
        title="Compute the closure of a subset in a linear matroid",
        description=(
            "Compute the closure (smallest flat) of a subset S in a matroid "
            "represented by the columns of a canonical matrix over GF(p). "
            "The closure adds all elements that lie in the span of S."
        ),
        request_type=MatroidClosureRequest,
        result_type=MatroidClosureResult,
        run=_run_closure,
        tags=("matroid", "closure", "flat", "exact"),
        examples=(
            OperationExample(
                name="closure_of_basis",
                description="Compute the closure of {0, 1} in a rank-2 matroid.",
                input=_CLOSURE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="matroid.basis.maximum_weight.compute",
        title="Compute a maximum-weight basis of a linear matroid",
        description=(
            "Run the deterministic greedy algorithm (decreasing weight, ties "
            "by increasing index) on a represented matroid with exact integer "
            "weights keyed by the exact ground axis. Return one maximum-weight "
            "basis, its exact total, the deterministic greedy order, and one "
            "fundamental-circuit exchange row per outside element replaying "
            "that no valid single-element exchange strictly improves the total."
        ),
        request_type=MaximumWeightBasisRequest,
        result_type=MaximumWeightBasisResult,
        run=_run_maximum_weight_basis,
        tags=("matroid", "basis", "maximum-weight", "greedy", "exact"),
        discovery_terms=(
            "maximum weight basis",
            "matroid greedy algorithm",
            "fundamental circuit",
            "basis exchange",
        ),
        examples=(
            OperationExample(
                name="triangle_weights_3_2_5",
                description="Maximum-weight basis of a rank-2 matroid picks {0, 2}.",
                input=_MAXIMUM_WEIGHT_BASIS_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="matroid.independent_set.maximum_weight.compute",
        title="Compute a maximum-weight independent set of a linear matroid",
        description=(
            "Run exact matroid greedy on a ground-axis-bound integer weight "
            "function. Positive-weight elements are scanned in decreasing "
            "weight order (ties by increasing ground index). "
            "Return one maximum-weight independent subset, exact total, "
            "rank, and the complete greedy consideration order. Unlike a "
            "basis operation, this may return a smaller set when weights "
            "are zero or negative."
        ),
        request_type=MaximumWeightIndependentSetRequest,
        result_type=MaximumWeightIndependentSetResult,
        run=_run_maximum_weight_independent_set,
        tags=("matroid", "independent-set", "maximum-weight", "greedy", "exact"),
        discovery_terms=(
            "maximum weight independent set",
            "maximum weight subset in a matroid",
            "negative matroid weights",
        ),
        examples=(
            OperationExample(
                name="negative_element_is_omitted",
                description="The negatively weighted element is not forced into an independent set.",
                input=_MAXIMUM_WEIGHT_INDEPENDENT_SET_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="matroid.intersection.weighted_certificate.check",
        title="Check a weighted common-independent-set optimality certificate",
        description=(
            "Check a caller-supplied common independent candidate and exact "
            "integer weight split w=u+v. Recompute each linear matroid's "
            "maximum split-weight independent-set value by its bounded greedy "
            "rank kernel; return the candidate and both source maxima only "
            "when the maxima sum to the candidate weight. This checks a "
            "certificate and does not search for a candidate or a split."
        ),
        request_type=MatroidWeightedIntersectionCertificateRequest,
        result_type=MatroidWeightedIntersectionResult,
        run=_run_weighted_intersection_certificate,
        tags=("matroid", "intersection", "weighted", "optimality", "exact"),
        discovery_terms=(
            "check a weighted matroid intersection certificate",
            "certify maximum weight common independent set",
            "Frank weight splitting witness for matroid intersection",
        ),
        examples=(
            OperationExample(
                name="parallel_pair_weighted_optimum",
                description=(
                    "Certify the weight-5 singleton in two identical rank-one "
                    "matroids using the split (5, 3) + (0, 0)."
                ),
                input={
                    "first": {
                        "matrix": {"prime": 2, "entries": [[1, 1]], "columns": 2},
                        "ground_labels": ["a", "b"],
                    },
                    "second": {
                        "matrix": {"prime": 2, "entries": [[1, 1]], "columns": 2},
                        "ground_labels": ["a", "b"],
                    },
                    "weight_function": {
                        "ground_axis": ["a", "b"],
                        "values": [5, 3],
                    },
                    "common_independent": [0],
                    "first_split": {
                        "ground_axis": ["a", "b"],
                        "values": [5, 3],
                    },
                    "second_split": {
                        "ground_axis": ["a", "b"],
                        "values": [0, 0],
                    },
                },
            ),
        ),
    ),
)
TOOLS = TOOLS + (  # noqa: RUF005
    MathTool(
        operation_id="matroid.intersection.compute",
        title="Compute a maximum common independent set",
        description=(
            "Compute an exact maximum-cardinality common independent set of "
            "two represented matroids on one labelled ground. Return both "
            "source ranks of the selected set and an Edmonds min-max rank "
            "witness; serialized claims replay all four ranks against their "
            "retained source matroids."
        ),
        request_type=MatroidIntersectionRequest,
        result_type=MatroidIntersectionResult,
        run=lambda request: matroid_intersection(request.first, request.second),
        tags=("matroid", "intersection", "exact"),
        examples=(
            OperationExample(
                name="same_triangle",
                description="Intersect two identical rank-two triangle matroids on the same labelled ground.",
                input={
                    "first": {
                        "matrix": {
                            "prime": 5,
                            "entries": [[1, 0, 1], [0, 1, 1]],
                            "columns": 3,
                        },
                        "ground_labels": ["a", "b", "c"],
                    },
                    "second": {
                        "matrix": {
                            "prime": 5,
                            "entries": [[1, 0, 1], [0, 1, 1]],
                            "columns": 3,
                        },
                        "ground_labels": ["a", "b", "c"],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matroid.intersection.common_basis.compute",
        title="Decide whether two linear matroids share a common basis",
        description=(
            "Compute the exact maximum common independent set, both full "
            "source ranks, and an Edmonds min-max rank witness. Return "
            "COMMON_BASIS only when the maximum set has both basis ranks; "
            "otherwise return NO_COMMON_BASIS with an explicit rank-mismatch "
            "or maximum-intersection-deficit reason."
        ),
        request_type=MatroidCommonBasisRequest,
        result_type=MatroidCommonBasisResult,
        run=_run_common_basis,
        tags=("matroid", "intersection", "common-basis", "exact"),
        discovery_terms=(
            "common basis of two matroids",
            "matroid common basis existence",
            "no common basis certificate",
            "Edmonds intersection min-max witness",
        ),
        examples=(
            OperationExample(
                name="same_triangle_has_a_common_basis",
                description="Two identical rank-two triangle matroids share any spanning pair.",
                input={
                    "first": {
                        "matrix": {
                            "prime": 5,
                            "entries": [[1, 0, 1], [0, 1, 1]],
                            "columns": 3,
                        },
                        "ground_labels": ["a", "b", "c"],
                    },
                    "second": {
                        "matrix": {
                            "prime": 5,
                            "entries": [[1, 0, 1], [0, 1, 1]],
                            "columns": 3,
                        },
                        "ground_labels": ["a", "b", "c"],
                    },
                },
            ),
            OperationExample(
                name="disjoint_unique_bases_no_common_basis",
                description="Two rank-one matroids with different unique nonloop elements cannot share a basis.",
                input={
                    "first": {
                        "matrix": {
                            "prime": 2,
                            "entries": [[1, 0]],
                            "columns": 2,
                        },
                        "ground_labels": ["a", "b"],
                    },
                    "second": {
                        "matrix": {
                            "prime": 2,
                            "entries": [[0, 1]],
                            "columns": 2,
                        },
                        "ground_labels": ["a", "b"],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
