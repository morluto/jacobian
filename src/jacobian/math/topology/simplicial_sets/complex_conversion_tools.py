"""Catalog declaration for finite complex conversion."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.operations import canonicalize
from jacobian.math.topology.simplicial_sets.complex_conversion import (
    simplicial_set_from_complex,
)
from jacobian.math.topology.simplicial_sets.complex_conversion_models import (
    SimplicialComplexPrefixRequest,
    SimplicialComplexPrefixResult,
)

_EDGE = canonicalize(("a", "b"), (("a", "b"),)).complex


def _convert(
    request: SimplicialComplexPrefixRequest,
) -> SimplicialComplexPrefixResult:
    return simplicial_set_from_complex(request)


TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.from_simplicial_complex.compute",
        title="Construct a simplicial-set prefix from a finite complex",
        description=(
            "Construct the associated simplicial-set prefix through the requested "
            "degree: simplices are monotone vertex tuples whose support is a face "
            "of the source complex; faces delete entries and degeneracies repeat "
            "entries. Return the checked FiniteTruncatedSimplicialSet and explicit "
            "indices for source faces represented nondegenerately in retained "
            "degrees. Per-degree simplex counts, aggregate simplices, map rows, "
            "identity-check work, transport rows, and output bytes are admitted "
            "before expansion."
        ),
        request_type=SimplicialComplexPrefixRequest,
        result_type=SimplicialComplexPrefixResult,
        run=_convert,
        tags=(
            "topology",
            "simplicial-set",
            "simplicial-complex",
            "conversion",
            "exact",
        ),
        discovery_terms=(
            "simplicial complex to simplicial set",
            "associated simplicial set",
            "monotone vertex tuples",
            "repeated-vertex simplex",
        ),
        examples=(
            OperationExample(
                name="edge_prefix_through_degree_two",
                description=(
                    "The edge produces two vertices and one nondegenerate edge, "
                    "along with all repeated-vertex simplices through degree two."
                ),
                input={"complex": _EDGE.model_dump(mode="json"), "max_degree": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
