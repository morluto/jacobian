"""Public declaration for finite simplicial-set truncation."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.standard import standard_simplex
from jacobian.math.topology.simplicial_sets.truncate import truncate_simplicial_set
from jacobian.math.topology.simplicial_sets.truncate_models import (
    SimplicialSetTruncateRequest,
)


def _truncate(
    request: SimplicialSetTruncateRequest,
) -> FiniteTruncatedSimplicialSet:
    return truncate_simplicial_set(request)


_INPUT = {
    "simplicial_set": standard_simplex(1, 2).model_dump(mode="json"),
    "max_degree": 1,
}

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.truncate.compute",
        title="Truncate a finite simplicial-set prefix",
        description=(
            "Retain the exact degrees 0..N and every face and degeneracy map "
            "visible there. N must not exceed the input maximum degree. "
            "Recheck all simplicial identities in the retained prefix and "
            "return it as a reusable finite truncated simplicial set."
        ),
        request_type=SimplicialSetTruncateRequest,
        result_type=FiniteTruncatedSimplicialSet,
        run=_truncate,
        tags=("topology", "simplicial-set", "truncation", "exact"),
        discovery_terms=(
            "truncate simplicial set",
            "simplicial set prefix",
            "degree truncated simplicial set",
        ),
        examples=(
            OperationExample(
                name="delta_one_prefix",
                description="Truncate Delta[1] through degree 2 to degrees 0 and 1.",
                input=_INPUT,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
