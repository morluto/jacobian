"""Independent checker declaration owned by the graph-symmetry domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.graph_symmetry import GraphSymmetryOrbitRequest

GRAPH_SYMMETRY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "graph.symmetry.generator_orbits.compute",
        GraphSymmetryOrbitRequest,
        "check_graph_symmetry_generator_orbits",
        "graph.symmetry.generator-orbits.stdlib-replay",
        entrypoint_module="jacobian_checkers.graph_exact_operations",
        replay_method="declared color-preserving generator orbit replay",
        reason=(
            "operator-authorized standard-library checker independently validates "
            "every declared automorphism and reconstructs both orbit partitions"
        ),
        verification_capability_id="graph.symmetry.generator_orbits.verify",
        verification_title="Verify declared graph-symmetry orbits",
        verification_description=(
            "Independently verify every declared color-preserving graph "
            "automorphism and replay its generated vertex and edge orbit partitions."
        ),
        verification_tags=(
            "verification",
            "exact",
            "graph",
            "symmetry",
            "automorphism",
            "orbit",
        ),
    ),
)

__all__ = ["GRAPH_SYMMETRY_EXACT_REPLAY_CHECKERS"]
