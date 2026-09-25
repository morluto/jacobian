# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import *
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapCompositionRequest,
    TruncatedSimplicialMap,
    compose_simplicial_maps,
    normalized_homology,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _map(r: Any) -> Any:
    return simplicial_map(r)


def _chains(r: Any) -> Any:
    return normalized_chains(r.simplicial_set)


def _homology(r: Any) -> Any:
    return normalized_homology(r.simplicial_set)


def _compose(r: SimplicialMapCompositionRequest) -> TruncatedSimplicialMap:
    return compose_simplicial_maps(r)


def _identity(r: FiniteTruncatedSimplicialSet) -> TruncatedSimplicialMap:
    return identity_simplicial_map(r)


_S = standard_simplex(1, 2).model_dump(mode="json")
_MAP = {"source": _S, "target": _S, "maps": [[0, 1], [0, 1, 2], [0, 1, 2, 3]]}
TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.map.compose.compute",
        title="Compose finite simplicial maps",
        description=(
            "Compose two degreewise maps between finite truncated simplicial "
            "sets. Recheck both maps against every visible face and degeneracy "
            "square, require exact middle-carrier identity, and return the "
            "degreewise composite as a simplicial-map value. The maps' carriers "
            "are bounded to degree 4 and 96 total simplices each."
        ),
        request_type=SimplicialMapCompositionRequest,
        result_type=TruncatedSimplicialMap,
        run=_compose,
        tags=("topology", "simplicial-set", "map", "composition", "exact"),
        examples=(
            OperationExample(
                name="compose_delta_one_identity_maps",
                description="Compose two identity maps on the same Delta[1] prefix.",
                input={
                    "first": {"source": _S, "target": _S, "maps": _MAP["maps"]},
                    "second": {"source": _S, "target": _S, "maps": _MAP["maps"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_map.compute",
        title="Check a finite simplicial map",
        description="Check degreewise simplex maps against every available face and degeneracy identity in two finite truncated simplicial-set carriers.",
        request_type=SimplicialMapRequest,
        result_type=SimplicialMapResult,
        run=_map,
        tags=("topology", "simplicial-set", "map", "exact"),
        examples=(
            OperationExample(
                name="identity_delta_one",
                description="Check the identity map of a finite Delta[1] prefix; source and target degree tables must match.",
                input=_MAP,
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_set.normalized_chains.compute",
        title="Compute a finite normalized-chain prefix",
        description=(
            "Remove in-range degeneracies from a finite simplicial-set prefix and "
            "return the normalized chain complex as the shared ChainComplexValue, "
            "with exact nondegenerate simplex labels for every degree axis. "
            "The alternating differential is checked to square to zero; the top "
            "group is retained without inferring a differential above the prefix."
        ),
        request_type=NormalizedChainsRequest,
        result_type=NormalizedChainsResult,
        run=_chains,
        tags=("topology", "simplicial-set", "normalized-chains", "exact"),
        examples=(
            OperationExample(
                name="normalized_delta_one",
                description="Compute normalized chains of a degree-0..2 Delta[1] prefix; the prefix retains all available degeneracies.",
                input={"simplicial_set": _S},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_set.homology.compute",
        title="Compute supported integral normalized homology",
        description=(
            "Compute exact integral homology from the normalized chains of a "
            "complete finite simplicial-set prefix. Return H_0 through "
            "H_(N-1), with normalized simplex axes and exact free and torsion "
            "representatives. H_N is omitted because the prefix does not "
            "contain its incoming boundary from degree N+1."
        ),
        request_type=NormalizedChainsRequest,
        result_type=NormalizedHomologyResult,
        run=_homology,
        tags=("topology", "simplicial-set", "homology", "normalized", "exact"),
        examples=(
            OperationExample(
                name="delta_one_integral_homology",
                description=(
                    "Compute H_0 and H_1 of a degree-0..2 Delta[1] prefix; "
                    "H_2 is omitted because d_3 is absent."
                ),
                input={"simplicial_set": _S},
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
