"""Tree-decomposition operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.tree_decompositions._models import (
    AdhesionsRequest,
    AdhesionsResult,
    BagIntersectionGraphRequest,
    BagIntersectionGraphResult,
    RerootRequest,
    RerootResult,
    RestrictRequest,
    RestrictResult,
    VertexOccurrencesRequest,
    VertexOccurrencesResult,
    WidthRequest,
    WidthResult,
)
from jacobian.math.graphs.tree_decompositions._operations import (
    compute_adhesions,
    compute_bag_intersection_graph,
    compute_reroot,
    compute_restrict,
    compute_vertex_occurrences,
    compute_width,
)


def _op[
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# A path graph a-b-c (two edges) with two bags: {a,b} and {b,c}.
_TN = ["t0", "t1"]
_GRAPH = {
    "graph_schema_version": "1",
    "vertices": ["a", "b", "c"],
    "edges": [["a", "b"], ["b", "c"]],
}
_DECOMPOSITION = {
    "graph": _GRAPH,
    "tree_nodes": ["t0", "t1"],
    "tree_edges": [["t0", "t1"]],
    "bags": [["a", "b"], ["b", "c"]],
}


TREE_DECOMPOSITION_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "graph.tree_decomposition.width.compute",
        "Compute the width of a tree decomposition",
        "Return bag cardinality per tree node, maximum bag cardinality, width "
        "(max bag cardinality minus one), and the maximum-bag node labels. The "
        "width of a decomposition supplies an upper bound on graph treewidth "
        "only.",
        WidthRequest,
        WidthResult,
        compute_width,
        "tree-decomposition",
        "width",
        "exact",
        examples=(
            example(
                "path_width_one",
                "Width of a path graph tree decomposition.",
                {"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
    _op(
        "graph.tree_decomposition.vertex_occurrences.compute",
        "Compute per-source-vertex occurrence subtrees",
        "Return the exact finite map source vertex -> connected subtree node "
        "set / induced tree edges, with occurrence counts and leaf/extremal "
        "nodes. Useful for decomposition-based constructions.",
        VertexOccurrencesRequest,
        VertexOccurrencesResult,
        compute_vertex_occurrences,
        "tree-decomposition",
        "vertex-occurrences",
        "exact",
        examples=(
            example(
                "path_vertex_occurrences",
                "Vertex occurrences of a path graph tree decomposition.",
                {"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
    _op(
        "graph.tree_decomposition.adhesions.compute",
        "Compute adhesions of a tree decomposition",
        "For every decomposition-tree edge tt', compute adhesion(t,t') = B_t "
        "intersection B_t', size, and the left/right component vertex coverage "
        "after deleting tt'. Return the maximum adhesion, size profile, and "
        "exact separator sets. A structural profile of the supplied "
        "decomposition, not a minimum-separator computation.",
        AdhesionsRequest,
        AdhesionsResult,
        compute_adhesions,
        "tree-decomposition",
        "adhesions",
        "exact",
        examples=(
            example(
                "path_adhesions",
                "Adhesions of a path graph tree decomposition.",
                {"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
    _op(
        "graph.tree_decomposition.reroot.compute",
        "Reroot a tree decomposition at a selected tree node",
        "Return the same underlying decomposition with a parent map, children "
        "map, depth per bag, and root-to-node paths. Changing the root does "
        "not change the width, bags, or unrooted tree.",
        RerootRequest,
        RerootResult,
        compute_reroot,
        "tree-decomposition",
        "reroot",
        "exact",
        examples=(
            example(
                "path_reroot_t1",
                "Reroot a path graph tree decomposition at t1.",
                {"decomposition": _DECOMPOSITION, "root": "t1"},
            ),
        ),
    ),
    _op(
        "graph.tree_decomposition.restrict.compute",
        "Restrict a tree decomposition to a source-vertex subset",
        "Return the decomposition obtained by replacing every bag B_t with "
        "B_t intersection S, then applying the documented deterministic "
        "cleanup of empty/redundant tree nodes. Bind the result to the induced "
        "source graph G[S]. A direct transformation, not a better-decomposition "
        "search.",
        RestrictRequest,
        RestrictResult,
        compute_restrict,
        "tree-decomposition",
        "restrict",
        "exact",
        examples=(
            example(
                "path_restrict_ab",
                "Restrict a path graph tree decomposition to {a,b}.",
                {"decomposition": _DECOMPOSITION, "subset": ["a", "b"]},
            ),
        ),
    ),
    _op(
        "graph.tree_decomposition.bag_intersection_graph.compute",
        "Compute the weighted bag-intersection graph",
        "Return the weighted tree itself with each edge labelled by its exact "
        "adhesion set/size and each node labelled by bag size. A compact "
        "projection useful for later structural summaries.",
        BagIntersectionGraphRequest,
        BagIntersectionGraphResult,
        compute_bag_intersection_graph,
        "tree-decomposition",
        "bag-intersection",
        "exact",
        examples=(
            example(
                "path_bag_intersection",
                "Bag-intersection graph of a path graph tree decomposition.",
                {"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
)

TOOLS = TREE_DECOMPOSITION_OPERATIONS

__all__ = ["TOOLS"]
