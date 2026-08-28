"""Exact structural graph decomposition operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.decomposition._models import (
    BiconnectedComponentsRequest,
    BiconnectedComponentsResult,
    BlockCutTreeRequest,
    BlockCutTreeResult,
    BridgeBlockRequest,
    BridgeBlockResult,
    EarDecompositionRequest,
    EarDecompositionResult,
    SPQRTreeRequest,
    SPQRTreeResult,
)
from jacobian.math.graphs.decomposition.operations import (
    biconnected_components,
    block_cut_tree,
    bridge_block_tree,
    ear_decomposition,
    spqr_tree,
)


def _compute_biconnected_components(
    request: BiconnectedComponentsRequest,
) -> BiconnectedComponentsResult:
    return biconnected_components(request.graph)


def _compute_block_cut_tree(request: BlockCutTreeRequest) -> BlockCutTreeResult:
    return block_cut_tree(request.graph)


def _compute_bridge_block_tree(request: BridgeBlockRequest) -> BridgeBlockResult:
    return bridge_block_tree(request.graph)


def _compute_ear_decomposition(
    request: EarDecompositionRequest,
) -> EarDecompositionResult:
    return ear_decomposition(request.graph)


def _compute_spqr_tree(request: SPQRTreeRequest) -> SPQRTreeResult:
    return spqr_tree(request.graph)


def graph_decomposition_operation[
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
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    graph_decomposition_operation(
        "graph.decomposition.spqr_tree.compute",
        "Compute a normalized SPQR tree of an undirected graph",
        "Compute a deterministic normalized S/P/Q/R decomposition of a"
        " biconnected finite simple graph. Real source edges occur in exactly"
        " one skeleton; paired virtual edges encode each separator gluing and"
        " the returned tree replays to exactly the source graph. A graph outside"
        " the biconnected minimum-size convention returns a concrete witness.",
        SPQRTreeRequest,
        SPQRTreeResult,
        _compute_spqr_tree,
        "graph",
        "decomposition",
        "spqr",
        "triconnected",
        "exact",
        examples=(
            example(
                "k4_rigid",
                "Compute the rigid SPQR skeleton of K4; the source graph is"
                " biconnected and has at least three vertices.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [
                            [0, 1],
                            [0, 2],
                            [0, 3],
                            [1, 2],
                            [1, 3],
                            [2, 3],
                        ],
                    }
                },
            ),
        ),
    ),
    graph_decomposition_operation(
        "graph.decomposition.block_cut_tree.compute",
        "Compute the block-cut tree of an undirected graph",
        "Decompose an undirected graph into its biconnected components (blocks) and articulation points using NetworkX. Returns the blocks, the articulation points, and the edges of the bipartite block-cut tree joining each block to the articulation points it contains.",
        BlockCutTreeRequest,
        BlockCutTreeResult,
        _compute_block_cut_tree,
        "graph",
        "decomposition",
        "block-cut",
        "exact",
        examples=(
            example(
                "two_triangle_blocks",
                "Compute the block-cut tree of a graph with two triangles sharing one vertex.",
                {
                    "graph": {
                        "vertex_count": 5,
                        "edges": [
                            [0, 1],
                            [1, 2],
                            [0, 2],
                            [0, 3],
                            [3, 4],
                            [0, 4],
                        ],
                    },
                },
            ),
        ),
    ),
    graph_decomposition_operation(
        "graph.decomposition.bridge_block_tree.compute",
        "Compute the bridge-block tree of an undirected graph",
        "Decompose an undirected graph into its 2-edge-connected components (bridge-blocks) using NetworkX. Returns the components, the bridges as normalised (u, v) pairs, and the edges of the bridge block tree joining adjacent components across each bridge.",
        BridgeBlockRequest,
        BridgeBlockResult,
        _compute_bridge_block_tree,
        "graph",
        "decomposition",
        "bridge",
        "exact",
        examples=(
            example(
                "two_triangles_bridge",
                "Compute the bridge-block tree of two triangles joined by a single bridge edge.",
                {
                    "graph": {
                        "vertex_count": 6,
                        "edges": [
                            [0, 1],
                            [1, 2],
                            [0, 2],
                            [3, 4],
                            [4, 5],
                            [3, 5],
                            [2, 3],
                        ],
                    },
                },
            ),
        ),
    ),
    graph_decomposition_operation(
        "graph.decomposition.ear.compute",
        "Compute an open ear decomposition of a biconnected graph",
        "Compute an open ear decomposition of a biconnected undirected graph. The first ear is a cycle and each subsequent ear is a path whose internal vertices are disjoint from all earlier ears. A graph that is not biconnected returns biconnected=false and no ears.",
        EarDecompositionRequest,
        EarDecompositionResult,
        _compute_ear_decomposition,
        "graph",
        "decomposition",
        "ear",
        "exact",
        examples=(
            example(
                "cycle_c4",
                "Compute an ear decomposition of a 4-cycle (a single ear).",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [
                            [0, 1],
                            [1, 2],
                            [2, 3],
                            [0, 3],
                        ],
                    },
                },
            ),
        ),
    ),
    graph_decomposition_operation(
        "graph.decomposition.biconnected_components.compute",
        "List the biconnected components of an undirected graph",
        "List all biconnected components of an undirected graph. Each component is returned as a sorted tuple of vertices.",
        BiconnectedComponentsRequest,
        BiconnectedComponentsResult,
        _compute_biconnected_components,
        "graph",
        "decomposition",
        "biconnected",
        "exact",
        examples=(
            example(
                "two_triangle_blocks",
                "List the biconnected components of two triangles sharing one vertex.",
                {
                    "graph": {
                        "vertex_count": 5,
                        "edges": [[0, 1], [1, 2], [0, 2], [0, 3], [3, 4], [0, 4]],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
