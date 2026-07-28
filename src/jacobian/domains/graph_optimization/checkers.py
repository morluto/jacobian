"""Independent checker declarations owned by the graph-optimization domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.graph_invariant_operations import GraphInvariantRequest
from jacobian.contracts.graph_optimization import GraphOptimizationRequest

_GRAPH_ENTRYPOINT = "jacobian_checkers.graph_exact_operations"

GRAPH_OPTIMIZATION_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "graph.induced_tree.maximum.compute",
        GraphOptimizationRequest,
        "check_graph_induced_tree_maximum",
        "graph.induced-tree.maximum.exhaustive-replay",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        replay_method="finite-subset exhaustive replay",
        reason=(
            "operator-authorized finite exhaustive checker independent of the "
            "Z3 producer"
        ),
        verification_capability_id="graph.induced_tree.maximum.verify",
        verification_title="Verify a maximum induced tree result",
        verification_description=(
            "Independently exhaust bounded vertex subsets to verify one stored "
            "exact maximum induced-tree result and its graph binding."
        ),
        verification_tags=("verification", "exact", "graph", "induced-tree"),
    ),
    ExactReplayCheckerDeclaration(
        "graph.invariant.maximum_matching.compute",
        GraphInvariantRequest,
        "check_graph_maximum_matching",
        "graph.maximum-matching.tutte-berge-v1",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        replay_method="Tutte-Berge barrier replay",
        reason=(
            "operator-authorized standard-library Tutte-Berge checker independent "
            "of the NetworkX producer"
        ),
        verification_capability_id="graph.invariant.maximum_matching.verify",
        verification_title="Verify a maximum matching result",
        verification_description=(
            "Independently verify matching feasibility and a Tutte-Berge upper-bound "
            "certificate for one exact stored finite graph."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "matching",
            "tutte-berge",
        ),
    ),
)


__all__ = ["GRAPH_OPTIMIZATION_EXACT_REPLAY_CHECKERS"]
