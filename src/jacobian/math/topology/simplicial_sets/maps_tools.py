# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.maps import *
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _map(r: Any) -> Any:
    return simplicial_map(r)


def _chains(r: Any) -> Any:
    return normalized_chains(r.simplicial_set)


_S = standard_simplex(1, 2).model_dump(mode="json")
_MAP = {"source": _S, "target": _S, "maps": [[0, 1], [0, 1, 2], [0, 1, 2, 3]]}
TOOLS = (
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
        description="Remove in-range degeneracies from a finite simplicial-set prefix, assemble the alternating normalized boundary matrices, and report the finite-prefix square-zero identity.",
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
)
__all__ = ["TOOLS"]
