"""Tree-decomposition operation declarations."""

from typing import Any

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.graphs.decomposition.tree_decompositions._models import (
    AdhesionsRequest,
    AdhesionsResult,
    BagIntersectionGraphRequest,
    BagIntersectionGraphResult,
    RerootRequest,
    RerootResult,
    RestrictRequest,
    VertexOccurrencesRequest,
    VertexOccurrencesResult,
    WidthRequest,
    WidthResult,
    _normalized_tree_nodes,
    _reroot_result_wire_bytes,
)
from jacobian.math.graphs.decomposition.tree_decompositions.operations import (
    adhesions,
    bag_intersection_graph,
    reroot,
    restrict,
    vertex_occurrences,
    width,
)
from jacobian.math.graphs.decomposition.tree_decompositions.values import (
    TreeDecomposition,
)


def compute_width(request: WidthRequest) -> WidthResult:
    return width(request.decomposition)


def compute_vertex_occurrences(
    request: VertexOccurrencesRequest,
) -> VertexOccurrencesResult:
    return vertex_occurrences(request.decomposition)


def compute_adhesions(request: AdhesionsRequest) -> AdhesionsResult:
    return adhesions(request.decomposition)


def compute_reroot(request: RerootRequest) -> RerootResult:
    if request.root not in request.decomposition.tree_nodes:
        raise OperationDomainValidationError(
            location=("root",),
            code="graph.root_must_be_a_declared_tree_node",
            message="root must be a declared tree node",
        )
    normalized_nodes = _normalized_tree_nodes(request.decomposition)
    if len(set(normalized_nodes)) != len(normalized_nodes):
        raise OperationDomainValidationError(
            location=("decomposition", "tree_nodes"),
            code="graph.reroot_tree_node_labels_collide_after_normalization",
            message="tree node labels collide after Unicode NFC normalization",
        )
    output_limit = CanonicalLimits().max_output_bytes
    if _reroot_result_wire_bytes(request.decomposition, request.root) > output_limit:
        raise OperationDomainValidationError(
            location=("decomposition", "tree_nodes"),
            code="graph.reroot_result_exceeds_transport_limit",
            message="rerooted tree-decomposition paths exceed the "
            f"{output_limit}-byte canonical output limit",
        )
    return reroot(request.decomposition, request.root)


def compute_restrict(request: RestrictRequest) -> TreeDecomposition:
    if not set(request.subset).issubset(request.decomposition.graph.vertices):
        raise OperationDomainValidationError(
            location=("subset",),
            code="graph.subset_must_contain_only_declared_source_vertice",
            message="subset must contain only declared source vertices",
        )
    return restrict(request.decomposition, frozenset(request.subset))


def compute_bag_intersection_graph(
    request: BagIntersectionGraphRequest,
) -> BagIntersectionGraphResult:
    return bag_intersection_graph(request.decomposition)


# A path graph a-b-c (two edges) with two bags: {a,b} and {b,c}.
_TN = ["t0", "t1"]
_GRAPH = {
    "vertices": ["a", "b", "c"],
    "edges": [["a", "b"], ["b", "c"]],
}
_DECOMPOSITION = {
    "graph": _GRAPH,
    "tree_nodes": ["t0", "t1"],
    "tree_edges": [["t0", "t1"]],
    "bags": [["a", "b"], ["b", "c"]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.tree_decomposition.width.compute",
        title="Compute the width of a tree decomposition",
        description="Return bag cardinality per tree node, maximum bag cardinality, width "
        "(max bag cardinality minus one), and the maximum-bag node labels. The "
        "width of a decomposition supplies an upper bound on graph treewidth "
        "only.",
        request_type=WidthRequest,
        result_type=WidthResult,
        run=compute_width,
        tags=("tree-decomposition", "width", "exact"),
        examples=(
            OperationExample(
                name="path_width_one",
                description="Width of a path graph tree decomposition.",
                input={"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.tree_decomposition.vertex_occurrences.compute",
        title="Compute per-source-vertex occurrence subtrees",
        description="Return the exact finite map source vertex -> connected subtree node "
        "set / induced tree edges, with occurrence counts and leaf/extremal "
        "nodes. Useful for decomposition-based constructions.",
        request_type=VertexOccurrencesRequest,
        result_type=VertexOccurrencesResult,
        run=compute_vertex_occurrences,
        tags=("tree-decomposition", "vertex-occurrences", "exact"),
        examples=(
            OperationExample(
                name="path_vertex_occurrences",
                description="Vertex occurrences of a path graph tree decomposition.",
                input={"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.tree_decomposition.adhesions.compute",
        title="Compute adhesions of a tree decomposition",
        description="For every decomposition-tree edge tt', compute adhesion(t,t') = B_t "
        "intersection B_t', size, and the left/right component vertex coverage "
        "after deleting tt'. Return the maximum adhesion, size profile, and "
        "exact separator sets. A structural profile of the supplied "
        "decomposition, not a minimum-separator computation.",
        request_type=AdhesionsRequest,
        result_type=AdhesionsResult,
        run=compute_adhesions,
        tags=("tree-decomposition", "adhesions", "exact"),
        examples=(
            OperationExample(
                name="path_adhesions",
                description="Adhesions of a path graph tree decomposition.",
                input={"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.tree_decomposition.reroot.compute",
        title="Reroot a tree decomposition at a selected tree node",
        description="Return the same underlying decomposition with a parent map, children "
        "map, depth per bag, and root-to-node paths. Changing the root does "
        "not change the width, bags, or unrooted tree.",
        request_type=RerootRequest,
        result_type=RerootResult,
        run=compute_reroot,
        tags=("tree-decomposition", "reroot", "exact"),
        examples=(
            OperationExample(
                name="path_reroot_t1",
                description="Reroot a path graph tree decomposition at t1.",
                input={"decomposition": _DECOMPOSITION, "root": "t1"},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.tree_decomposition.restrict.compute",
        title="Restrict a tree decomposition to a source-vertex subset",
        description="Return the decomposition obtained by replacing every bag B_t with "
        "B_t intersection S, then applying the documented deterministic "
        "cleanup of empty/redundant tree nodes. Bind the result to the induced "
        "source graph G[S]. A direct transformation, not a better-decomposition "
        "search.",
        request_type=RestrictRequest,
        result_type=TreeDecomposition,
        run=compute_restrict,
        tags=("tree-decomposition", "restrict", "exact"),
        examples=(
            OperationExample(
                name="path_restrict_ab",
                description="Restrict a path graph tree decomposition to {a,b}.",
                input={"decomposition": _DECOMPOSITION, "subset": ["a", "b"]},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.tree_decomposition.bag_intersection_graph.compute",
        title="Compute the weighted bag-intersection graph",
        description="Return the weighted tree itself with each edge labelled by its exact "
        "adhesion set/size and each node labelled by bag size. A compact "
        "projection useful for later structural summaries.",
        request_type=BagIntersectionGraphRequest,
        result_type=BagIntersectionGraphResult,
        run=compute_bag_intersection_graph,
        tags=("tree-decomposition", "bag-intersection", "exact"),
        examples=(
            OperationExample(
                name="path_bag_intersection",
                description="Bag-intersection graph of a path graph tree decomposition.",
                input={"decomposition": _DECOMPOSITION},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
