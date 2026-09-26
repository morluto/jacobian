"""Public declaration for unnormalized simplicial chains."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.chains import (
    UnnormalizedChainsRequest,
    UnnormalizedChainsResult,
    unnormalized_chains,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _run(request: UnnormalizedChainsRequest) -> UnnormalizedChainsResult:
    return unnormalized_chains(request)


_DELTA_ONE = standard_simplex(1, 2).model_dump(mode="json")

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.unnormalized_chain_complex.compute",
        title="Compute the finite unnormalized simplicial chain complex",
        description=(
            "For a complete finite table-based simplicial-set prefix X_0..X_N, "
            "return the exact chain complex C_n=R[X_n] over ZZ, QQ, or a bounded "
            "prime field, with alternating face differential, source simplex "
            "axes, and the canonical ChainComplexValue. The top group is "
            "retained; no differential from an absent degree N+1 is inferred."
        ),
        request_type=UnnormalizedChainsRequest,
        result_type=UnnormalizedChainsResult,
        run=_run,
        tags=("topology", "simplicial-set", "chain-complex", "unnormalized", "exact"),
        discovery_terms=(
            "unnormalized simplicial chains",
            "alternating face boundary",
            "simplicial set chain complex",
        ),
        examples=(
            OperationExample(
                name="delta_one_unnormalized_chains",
                description=(
                    "The degree-0..2 prefix of Delta[1] has unnormalized ranks "
                    "2, 3, 4, including its degenerate simplices."
                ),
                input={"simplicial_set": _DELTA_ONE},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
