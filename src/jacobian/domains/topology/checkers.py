"""Independent checker declarations owned by the topology domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.topology import (
    ChainComplexRequest,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
)

_ENTRYPOINT = "jacobian_checkers.simplicial_topology"
_REASON = (
    "operator-authorized clean-process modular replay that imports no topology "
    "producer or contract implementation"
)

TOPOLOGY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "topology.simplicial_complex.materialize",
        SimplicialComplexRequest,
        "check_simplicial_complex_materialization",
        "topology.simplicial-complex.closure-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent finite face-closure replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "topology.simplicial_complex.chain_complex.compute",
        ChainComplexRequest,
        "check_simplicial_chain_complex",
        "topology.simplicial-chain.boundary-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent oriented-boundary replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "topology.simplicial_homology.compute",
        SimplicialHomologyRequest,
        "check_simplicial_homology",
        "topology.simplicial-homology.modular-replay",
        entrypoint_module=_ENTRYPOINT,
        replay_method="independent modular quotient replay",
        reason=_REASON,
    ),
)

__all__ = ["TOPOLOGY_EXACT_REPLAY_CHECKERS"]
