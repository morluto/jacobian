"""Rooted-tree operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.rooted_trees._models import (
    RootedTreeFinePartitionRequest,
)
from jacobian.math.graphs.rooted_trees.operations import construct_fine_partition
from jacobian.math.graphs.rooted_trees.values import RootedTreeFinePartition


def _run_fine_partition(
    request: RootedTreeFinePartitionRequest,
) -> RootedTreeFinePartition:
    return construct_fine_partition(
        request.graph, request.root, request.component_size_limit
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.rooted_tree.fine_partition.construct",
        title="Construct a rooted-tree fine partition",
        description=(
            "Construct a deterministic fine partition of a rooted finite simple "
            "tree into even- and odd-distance seed classes and bounded connected "
            "shrubs. Every shrub has a nonempty boundary wholly in one seed "
            "class, and each seed class has size at most "
            "12 * (n - 1) / component_size_limit. The result retains and exactly "
            "reconstructs the source graph; a non-tree returns a typed diagnostic. "
            "Rows are lexically ordered and shrubs are indexed rootward-first, "
            "with no label-independent canonicity claim."
        ),
        request_type=RootedTreeFinePartitionRequest,
        result_type=RootedTreeFinePartition,
        run=_run_fine_partition,
        tags=("graph", "rooted-tree", "fine-partition", "decomposition", "exact"),
        discovery_terms=(
            "fine partition",
            "rooted tree partition",
            "parity refined tree separator",
            "bounded shrubs",
        ),
        examples=(
            example(
                "path_fine_partition",
                (
                    "Construct a 2-fine partition of a five-vertex rooted path; "
                    "the supplied graph must be a tree, and "
                    "component_size_limit=2 bounds every returned shrub."
                ),
                {
                    "graph": {
                        "vertices": ["a", "b", "c", "d", "e"],
                        "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]],
                    },
                    "root": "a",
                    "component_size_limit": 2,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
