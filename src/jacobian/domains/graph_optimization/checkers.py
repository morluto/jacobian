"""Independent checker declarations owned by the graph-optimization domain."""

from collections.abc import Callable

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.graph_distance_matrix import GraphDistanceMatrixRequest
from jacobian.contracts.graph_invariant_operations import (
    GraphInvariantRequest,
    GraphMaximumMatchingRequest,
)
from jacobian.contracts.graph_optimization import (
    GraphHamiltonianPathRequest,
    GraphMinimumSpanningTreeRequest,
    GraphOptimizationRequest,
)
from jacobian.contracts.operations import ProviderObservation
from jacobian.providers import flint_runtime

_GRAPH_ENTRYPOINT = "jacobian_checkers.graph_exact_operations"


def _graph_runtime(*, checker_ids: tuple[str, ...] = ()) -> ProviderObservation:
    return flint_runtime.graph_exact_checker_provider_runtime(checker_ids=checker_ids)


def _graph_order_at_most(maximum: int) -> Callable[[object], bool]:
    def supports(payload: object) -> bool:
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("graph"), dict)
            and isinstance(payload["graph"].get("vertices"), list)
            and len(payload["graph"]["vertices"]) <= maximum
        )

    return supports


GRAPH_OPTIMIZATION_AUTHORIZED_CHECKERS = (
    AuthorizedChecker(
        "graph.hamiltonian_path.decide",
        GraphHamiltonianPathRequest,
        "check_graph_hamiltonian_path",
        "graph.hamiltonian-path.exhaustive-replay",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="finite Hamiltonian-path exhaustive replay",
        reason=(
            "operator-authorized standard-library checker independent of the "
            "producer's dynamic-programming implementation"
        ),
        verification_operation_id="graph.hamiltonian_path.verify",
        verification_title="Verify a Hamiltonian-path decision",
        verification_description=(
            "Independently verify a spanning path witness or exhaust the bounded "
            "finite path state space for one submitted negative decision."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "hamiltonian-path",
        ),
        supports_input=_graph_order_at_most(18),
    ),
    AuthorizedChecker(
        "graph.induced_tree.maximum.compute",
        GraphOptimizationRequest,
        "check_graph_induced_tree_maximum",
        "graph.induced-tree.maximum.exhaustive-replay",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="finite-subset exhaustive replay",
        reason=(
            "operator-authorized finite exhaustive checker independent of the "
            "Z3 producer"
        ),
        verification_operation_id="graph.induced_tree.maximum.verify",
        verification_title="Verify a maximum induced tree result",
        verification_description=(
            "Independently exhaust bounded vertex subsets to verify one submitted "
            "exact maximum induced-tree result against its exact graph input."
        ),
        verification_tags=("verification", "exact", "graph", "induced-tree"),
        supports_input=_graph_order_at_most(16),
    ),
    AuthorizedChecker(
        "graph.spanning_tree.minimum.compute",
        GraphMinimumSpanningTreeRequest,
        "check_graph_minimum_spanning_tree",
        "graph.minimum-spanning-tree.cycle-certificate-v1",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="fundamental-cycle optimality certificate replay",
        reason=(
            "operator-authorized standard-library exact-rational checker "
            "independent of the NetworkX Kruskal producer"
        ),
        verification_operation_id="graph.spanning_tree.minimum.verify",
        verification_title="Verify a weighted minimum spanning tree",
        verification_description=(
            "Independently verify source connectivity, spanning-tree feasibility, "
            "exact total weight, and every fundamental-cycle non-improvement check "
            "for one submitted exact rational weighted-graph result."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "weighted-graph",
            "minimum-spanning-tree",
            "cycle-property",
        ),
    ),
    AuthorizedChecker(
        "graph.invariant.diameter.compute",
        GraphInvariantRequest,
        "check_graph_diameter",
        "graph.diameter.all-sources-bfs-v1",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="all-sources breadth-first replay",
        reason=(
            "operator-authorized standard-library BFS checker independent of "
            "the NetworkX producer"
        ),
        verification_operation_id="graph.invariant.diameter.verify",
        verification_title="Verify an exact graph diameter",
        verification_description=(
            "Independently replay all-source shortest paths to verify one submitted "
            "diameter result, including its disconnected-graph convention."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "diameter",
            "breadth-first-search",
        ),
    ),
    AuthorizedChecker(
        "graph.invariant.radius.compute",
        GraphInvariantRequest,
        "check_graph_radius",
        "graph.radius.all-sources-bfs-v1",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="all-sources breadth-first replay",
        reason=(
            "operator-authorized standard-library BFS checker independent of "
            "the NetworkX producer"
        ),
        verification_operation_id="graph.invariant.radius.verify",
        verification_title="Verify an exact graph radius",
        verification_description=(
            "Independently replay all-source shortest paths to verify one submitted "
            "radius result, including its disconnected-graph convention."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "radius",
            "breadth-first-search",
        ),
    ),
    AuthorizedChecker(
        "graph.distance_matrix.compute",
        GraphDistanceMatrixRequest,
        "check_graph_distance_matrix",
        "graph.distance-matrix.all-sources-bfs-v1",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="all-sources breadth-first distance-matrix replay",
        reason=(
            "operator-authorized standard-library BFS checker independent of "
            "the NetworkX producer"
        ),
        verification_operation_id="graph.distance_matrix.verify",
        verification_title="Verify an exact graph distance matrix",
        verification_description=(
            "Independently replay every source shortest-path traversal to verify "
            "one submitted all-pairs distance matrix, including unreachable pairs."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "distance",
            "matrix",
            "breadth-first-search",
        ),
    ),
    AuthorizedChecker(
        "graph.invariant.maximum_matching.compute",
        GraphMaximumMatchingRequest,
        "check_graph_maximum_matching",
        "graph.maximum-matching.tutte-berge-v1",
        entrypoint_module=_GRAPH_ENTRYPOINT,
        observation_loader=_graph_runtime,
        replay_method="Tutte-Berge barrier replay",
        reason=(
            "operator-authorized standard-library Tutte-Berge checker independent "
            "of the NetworkX producer"
        ),
        verification_operation_id="graph.invariant.maximum_matching.verify",
        verification_title="Verify a maximum matching result",
        verification_description=(
            "Independently verify matching feasibility and a Tutte-Berge upper-bound "
            "certificate submitted with its exact finite graph input."
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

GRAPH_SEARCH_AUTHORIZED_CHECKERS = GRAPH_OPTIMIZATION_AUTHORIZED_CHECKERS[:3]
GRAPH_INVARIANT_AUTHORIZED_CHECKERS = GRAPH_OPTIMIZATION_AUTHORIZED_CHECKERS[3:]

__all__ = [
    "GRAPH_INVARIANT_AUTHORIZED_CHECKERS",
    "GRAPH_OPTIMIZATION_AUTHORIZED_CHECKERS",
    "GRAPH_SEARCH_AUTHORIZED_CHECKERS",
]
