"""Incidence structure operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    discovery_terms: tuple[str, ...] = (),
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
        discovery_terms=discovery_terms,
        examples=examples,
    )


_STRUCTURE = {
    "points": ["p1", "p2", "p3"],
    "block_ids": ["b1", "b2"],
    "blocks": [["p1", "p2"], ["p2", "p3"]],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "incidence.matrix.compute",
        "Compute the incidence matrix",
        "Compute the exact 0/1 incidence matrix of a finite incidence "
        "structure, with labelled point rows and block columns.",
        IncidenceMatrixRequest,
        IncidenceMatrixResult,
        _incidence_matrix,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_structure",
                "Compute the incidence matrix of a 3-point, 2-block structure; "
                "every block member must be a declared point.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.degree_profile.compute",
        "Compute point and block degree profiles",
        "Compute per-point degrees (number of blocks containing each point) "
        "and per-block degrees (number of points in each block), with total "
        "incidence count.",
        IncidenceMatrixRequest,
        DegreeProfileResult,
        _degree_profile,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_degrees",
                "Compute degree profiles for a 3-point, 2-block structure; "
                "every block member must be a declared point.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.containment_profiles.compute",
        "Compute t-subset containment multiplicity profiles",
        "For a bounded order t, return the finite map from every t-subset "
        "of points to the number of blocks containing it, plus the "
        "multiplicity histogram and whether the profile is constant.",
        ContainmentProfileRequest,
        ContainmentProfileResult,
        _containment_profile,
        "combinatorics",
        "incidence",
        "exact",
        discovery_terms=("t-codegree", "codegree profile"),
        examples=(
            example(
                "triangle_pair_codegrees",
                "Compute the exact t=2 codegrees of all pairs in a 3-point, "
                "2-block incidence structure, including the zero codegree of "
                "the pair {p1, p3}.",
                {"incidence": _STRUCTURE, "t": 2},
            ),
        ),
    ),
    _op(
        "incidence.trade.check",
        "Compare finite incidence trade moments",
        "Compare two indexed finite block families on the same ordered point "
        "axis through a requested maximum subset order, admitted order by "
        "order from the subset-count, work, and output budgets. Return "
        "per-order totals and every nonzero subset-multiplicity difference; "
        "omitted subsets have equal, possibly zero, multiplicity. Report the "
        "zeroth block-count difference separately.",
        IncidenceTradeRequest,
        IncidenceTradeResult,
        _incidence_trade,
        "combinatorics",
        "incidence",
        "design-trade",
        "exact",
        examples=(
            example(
                "unequal_block_count_with_equal_point_moments",
                "Compare {a},{b} with {a,b} through order one on the exact "
                "same point axis; the point multiplicities agree while the "
                "left family has one more indexed block.",
                {
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
    _op(
        "incidence.intersections.compute",
        "Compute block intersection profiles",
        "For every unordered pair of indexed blocks, return the intersection "
        "subset and cardinality, plus the intersection-size histogram.",
        IntersectionsRequest,
        IntersectionsResult,
        _intersections,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_intersections",
                "Compute pairwise block intersections for a 3-point, 2-block "
                "structure.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.dual.compute",
        "Compute the dual incidence structure",
        "Swap the point and indexed-block domains: dual points are the "
        "original block IDs and dual blocks are one per original point.",
        DualRequest,
        DualResult,
        _dual,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_dual",
                "Compute the dual of a 3-point, 2-block structure; dual "
                "points are the original block IDs.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.complement.compute",
        "Compute the complement incidence structure",
        "Replace every block by its complement in the same point domain, "
        "preserving block IDs and returning the exact old/new correspondence.",
        ComplementRequest,
        ComplementResult,
        _complement,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_complement",
                "Compute the block complement of a 3-point, 2-block "
                "structure; each block maps to its point complement.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.restriction.compute",
        "Compute point/block deletion and restriction",
        "Restrict to a supplied point subset and/or block subset.  Each "
        "block is intersected with the retained point domain; block IDs "
        "are preserved.",
        RestrictionRequest,
        RestrictionResult,
        _restriction,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_restriction",
                "Restrict a 3-point, 2-block structure to two points; "
                "blocks are intersected with the retained domain.",
                {
                    "incidence": _STRUCTURE,
                    "points": ["p1", "p2"],
                },
            ),
        ),
    ),
    _op(
        "incidence.derived_residual.compute",
        "Compute derived and residual incidence structures",
        "At a selected point p, return the derived structure (blocks "
        "containing p, with p removed) or the residual structure (blocks "
        "not containing p) on P \\ {p}.",
        DerivedResidualRequest,
        DerivedResidualResult,
        _derived_residual,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_derived",
                "Compute the derived incidence structure at point p2 of a "
                "3-point, 2-block structure.",
                {"incidence": _STRUCTURE, "point": "p2"},
            ),
        ),
    ),
    _op(
        "incidence.levi_graph.compute",
        "Compute the Levi graph",
        "Return the labelled bipartite incidence graph: left vertices are "
        "tagged point IDs and right vertices are tagged block IDs, with an "
        "edge for each incidence.",
        LeviGraphRequest,
        LeviGraphResult,
        _levi_graph,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_levi",
                "Compute the Levi graph of a 3-point, 2-block structure; "
                "point and block labels use distinct tagged namespaces.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.gram.compute",
        "Compute the Gram / concordance matrix",
        "From the labelled incidence matrix N, return the exact labelled "
        "integer Gram matrix N N^T (point axis) or N^T N (block axis).",
        GramRequest,
        GramResult,
        _gram,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_gram",
                "Compute the point-axis Gram matrix N N^T of a 3-point, "
                "2-block structure.",
                {"incidence": _STRUCTURE, "axis": "point"},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
