"""Incidence structure operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    ComplementRequest,
    ComplementResult,
    ContainmentProfileRequest,
    ContainmentProfileResult,
    DegreeProfileResult,
    DerivedResidualRequest,
    DerivedResidualResult,
    DualRequest,
    DualResult,
    GramRequest,
    GramResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    IncidenceTradeRequest,
    IncidenceTradeResult,
    IntersectionsRequest,
    IntersectionsResult,
    LeviGraphRequest,
    LeviGraphResult,
    RestrictionRequest,
    RestrictionResult,
)
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    check_incidence_trade,
    complement,
    containment_profile,
    degree_profile,
    derived_residual,
    dual,
    gram,
    incidence_matrix,
    intersections,
    levi_graph,
    restriction,
)


def _incidence_matrix(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    return incidence_matrix(request.incidence)


def _degree_profile(request: IncidenceMatrixRequest) -> DegreeProfileResult:
    return degree_profile(request.incidence)


def _containment_profile(
    request: ContainmentProfileRequest,
) -> ContainmentProfileResult:
    return containment_profile(request.incidence, request.t)


def _incidence_trade(request: IncidenceTradeRequest) -> IncidenceTradeResult:
    return check_incidence_trade(request.left, request.right, request.max_order)


def _intersections(request: IntersectionsRequest) -> IntersectionsResult:
    return intersections(request.incidence)


def _dual(request: DualRequest) -> DualResult:
    return dual(request.incidence)


def _complement(request: ComplementRequest) -> ComplementResult:
    return complement(request.incidence)


def _restriction(request: RestrictionRequest) -> RestrictionResult:
    return restriction(request.incidence, request.points, request.block_ids)


def _derived_residual(request: DerivedResidualRequest) -> DerivedResidualResult:
    return derived_residual(request.incidence, request.point, request.kind)


def _levi_graph(request: LeviGraphRequest) -> LeviGraphResult:
    return levi_graph(request.incidence)


def _gram(request: GramRequest) -> GramResult:
    return gram(request.incidence, request.axis)


_STRUCTURE = {
    "points": ["p1", "p2", "p3"],
    "block_ids": ["b1", "b2"],
    "blocks": [["p1", "p2"], ["p2", "p3"]],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="incidence.matrix.compute",
        title="Compute the incidence matrix",
        description="Compute the exact 0/1 incidence matrix of a finite incidence "
        "structure, with labelled point rows and block columns.",
        request_type=IncidenceMatrixRequest,
        result_type=IncidenceMatrixResult,
        run=_incidence_matrix,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_structure",
                description="Compute the incidence matrix of a 3-point, 2-block structure; "
                "every block member must be a declared point.",
                input={"incidence": _STRUCTURE},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.degree_profile.compute",
        title="Compute point and block degree profiles",
        description="Compute per-point degrees (number of blocks containing each point) "
        "and per-block degrees (number of points in each block), with total "
        "incidence count.",
        request_type=IncidenceMatrixRequest,
        result_type=DegreeProfileResult,
        run=_degree_profile,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_degrees",
                description="Compute degree profiles for a 3-point, 2-block structure; "
                "every block member must be a declared point.",
                input={"incidence": _STRUCTURE},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.containment_profiles.compute",
        title="Compute t-subset containment multiplicity profiles",
        description="For a bounded order t, return the finite map from every t-subset "
        "of points to the number of blocks containing it, plus the "
        "multiplicity histogram and whether the profile is constant.",
        request_type=ContainmentProfileRequest,
        result_type=ContainmentProfileResult,
        run=_containment_profile,
        tags=("combinatorics", "incidence", "exact"),
        discovery_terms=("t-codegree", "codegree profile"),
        examples=(
            OperationExample(
                name="triangle_pair_codegrees",
                description="Compute the exact t=2 codegrees of all pairs in a 3-point, "
                "2-block incidence structure, including the zero codegree of "
                "the pair {p1, p3}.",
                input={"incidence": _STRUCTURE, "t": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.trade.check",
        title="Compare finite incidence trade moments",
        description="Compare two indexed finite block families on the same ordered point "
        "axis through a requested maximum subset order, admitted order by "
        "order from the subset-count, work, and output budgets. Return "
        "per-order totals and every nonzero subset-multiplicity difference; "
        "omitted subsets have equal, possibly zero, multiplicity. Report the "
        "zeroth block-count difference separately.",
        request_type=IncidenceTradeRequest,
        result_type=IncidenceTradeResult,
        run=_incidence_trade,
        tags=("combinatorics", "incidence", "design-trade", "exact"),
        examples=(
            OperationExample(
                name="unequal_block_count_with_equal_point_moments",
                description="Compare {a},{b} with {a,b} through order one on the exact "
                "same point axis; the point multiplicities agree while the "
                "left family has one more indexed block.",
                input={
                    "left": {
                        "points": ["a", "b"],
                        "block_ids": ["l0", "l1"],
                        "blocks": [["a"], ["b"]],
                    },
                    "right": {
                        "points": ["a", "b"],
                        "block_ids": ["r0"],
                        "blocks": [["a", "b"]],
                    },
                    "max_order": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.intersections.compute",
        title="Compute block intersection profiles",
        description="For every unordered pair of indexed blocks, return the intersection "
        "subset and cardinality, plus the intersection-size histogram.",
        request_type=IntersectionsRequest,
        result_type=IntersectionsResult,
        run=_intersections,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_intersections",
                description="Compute pairwise block intersections for a 3-point, 2-block "
                "structure.",
                input={"incidence": _STRUCTURE},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.dual.compute",
        title="Compute the dual incidence structure",
        description="Swap the point and indexed-block domains: dual points are the "
        "original block IDs and dual blocks are one per original point.",
        request_type=DualRequest,
        result_type=DualResult,
        run=_dual,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_dual",
                description="Compute the dual of a 3-point, 2-block structure; dual "
                "points are the original block IDs.",
                input={"incidence": _STRUCTURE},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.complement.compute",
        title="Compute the complement incidence structure",
        description="Replace every block by its complement in the same point domain, "
        "preserving block IDs and returning the exact old/new correspondence.",
        request_type=ComplementRequest,
        result_type=ComplementResult,
        run=_complement,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_complement",
                description="Compute the block complement of a 3-point, 2-block "
                "structure; each block maps to its point complement.",
                input={"incidence": _STRUCTURE},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.restriction.compute",
        title="Compute point/block deletion and restriction",
        description="Restrict to a supplied point subset and/or block subset.  Each "
        "block is intersected with the retained point domain; block IDs "
        "are preserved.",
        request_type=RestrictionRequest,
        result_type=RestrictionResult,
        run=_restriction,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_restriction",
                description="Restrict a 3-point, 2-block structure to two points; "
                "blocks are intersected with the retained domain.",
                input={
                    "incidence": _STRUCTURE,
                    "points": ["p1", "p2"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.derived_residual.compute",
        title="Compute derived and residual incidence structures",
        description="At a selected point p, return the derived structure (blocks "
        "containing p, with p removed) or the residual structure (blocks "
        "not containing p) on P \\ {p}.",
        request_type=DerivedResidualRequest,
        result_type=DerivedResidualResult,
        run=_derived_residual,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_derived",
                description="Compute the derived incidence structure at point p2 of a "
                "3-point, 2-block structure.",
                input={"incidence": _STRUCTURE, "point": "p2"},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.levi_graph.compute",
        title="Compute the Levi graph",
        description="Return the labelled bipartite incidence graph: left vertices are "
        "tagged point IDs and right vertices are tagged block IDs, with an "
        "edge for each incidence.",
        request_type=LeviGraphRequest,
        result_type=LeviGraphResult,
        run=_levi_graph,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_levi",
                description="Compute the Levi graph of a 3-point, 2-block structure; "
                "point and block labels use distinct tagged namespaces.",
                input={"incidence": _STRUCTURE},
            ),
        ),
    ),
    MathTool(
        operation_id="incidence.gram.compute",
        title="Compute the Gram / concordance matrix",
        description="From the labelled incidence matrix N, return the exact labelled "
        "integer Gram matrix N N^T (point axis) or N^T N (block axis).",
        request_type=GramRequest,
        result_type=GramResult,
        run=_gram,
        tags=("combinatorics", "incidence", "exact"),
        examples=(
            OperationExample(
                name="triangle_gram",
                description="Compute the point-axis Gram matrix N N^T of a 3-point, "
                "2-block structure.",
                input={"incidence": _STRUCTURE, "axis": "point"},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
