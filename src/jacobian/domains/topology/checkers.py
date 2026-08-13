"""Independent checker declarations owned by the topology domain."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.topology import (
    ChainComplexRequest,
    IntegralSimplicialHomologyRequest,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
)
from jacobian.providers import flint_runtime

_ENTRYPOINT = "jacobian_checkers.simplicial_topology"
_REASON = (
    "operator-authorized clean-process modular replay that imports no topology "
    "producer or contract implementation"
)


def _topology_runtime(
    *, checker_ids: tuple[str, ...] = ()
) -> CapabilityProviderRuntime:
    return flint_runtime.topology_exact_checker_provider_runtime(
        checker_ids=checker_ids
    )


TOPOLOGY_EXACT_REPLAY_CHECKERS = (
    ExactReplayCheckerDeclaration(
        "topology.simplicial_complex.canonicalize",
        SimplicialComplexRequest,
        "check_simplicial_complex_canonicalization",
        "topology.simplicial-complex.closure-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_topology_runtime,
        replay_method="independent finite face-closure replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "topology.simplicial_complex.chain_complex.compute",
        ChainComplexRequest,
        "check_simplicial_chain_complex",
        "topology.simplicial-chain.boundary-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_topology_runtime,
        replay_method="independent oriented-boundary replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "topology.simplicial_homology.compute",
        SimplicialHomologyRequest,
        "check_simplicial_homology",
        "topology.simplicial-homology.modular-replay",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_topology_runtime,
        replay_method="independent modular quotient replay",
        reason=_REASON,
    ),
    ExactReplayCheckerDeclaration(
        "topology.simplicial_homology.integral.compute",
        IntegralSimplicialHomologyRequest,
        "check_integral_simplicial_homology",
        "topology.simplicial-homology.integral-smith-certificate-v1",
        entrypoint_module=_ENTRYPOINT,
        provider_runtime_factory=_topology_runtime,
        replay_method=(
            "independent integral chain and Smith transformation-certificate replay"
        ),
        reason=(
            "operator-authorized clean-process checker reconstructs every integer "
            "boundary, validates both Smith certificates per dimension, and binds "
            "free and torsion generators to the canonical simplex bases"
        ),
        verification_capability_id="topology.simplicial_homology.integral.verify",
        verification_title="Verify integral simplicial homology",
        verification_description=(
            "Independently verify every free rank, torsion factor, cycle generator, "
            "bounding chain, and embedded Smith transformation certificate."
        ),
        verification_tags=(
            "verification",
            "exact",
            "topology",
            "simplicial-homology",
            "integer",
            "torsion",
            "smith-normal-form",
        ),
    ),
)

__all__ = ["TOPOLOGY_EXACT_REPLAY_CHECKERS"]
