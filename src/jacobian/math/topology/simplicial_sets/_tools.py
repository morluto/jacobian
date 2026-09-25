"""Public declaration for finite truncated simplicial sets."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets._models import (
    SimplicialSetTablesRequest,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.chains_tools import TOOLS as CHAIN_TOOLS
from jacobian.math.topology.simplicial_sets.complex_conversion_tools import (
    TOOLS as COMPLEX_CONVERSION_TOOLS,
)
from jacobian.math.topology.simplicial_sets.coproduct_tools import (
    TOOLS as COPRODUCT_TOOLS,
)
from jacobian.math.topology.simplicial_sets.degeneracy_tools import (
    TOOLS as DEGENERACY_TOOLS,
)
from jacobian.math.topology.simplicial_sets.image_tools import TOOLS as IMAGE_TOOLS
from jacobian.math.topology.simplicial_sets.maps_tools import TOOLS as MAP_TOOLS
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.product_tools import TOOLS as PRODUCT_TOOLS
from jacobian.math.topology.simplicial_sets.skeleton_tools import (
    TOOLS as SKELETON_TOOLS,
)
from jacobian.math.topology.simplicial_sets.standard_tools import (
    TOOLS as STANDARD_TOOLS,
)
from jacobian.math.topology.simplicial_sets.subset_tools import TOOLS as SUBSET_TOOLS
from jacobian.math.topology.simplicial_sets.truncate_tools import (
    TOOLS as TRUNCATE_TOOLS,
)


def _run_from_tables(request: SimplicialSetTablesRequest) -> SimplicialSetTablesResult:
    return from_tables(
        request.max_degree, request.sets, request.face_maps, request.degeneracy_maps
    )


# Delta[1] truncated to degrees 0..1: X_0 = {0, 1}, X_1 = {00, 01, 11};
# faces delete one entry and s_0 duplicates the entry.
_DELTA_ONE_PREFIX = {
    "max_degree": 1,
    "sets": [["0", "1"], ["00", "01", "11"]],
    "face_maps": [[[0, 1, 1], [0, 0, 1]]],
    "degeneracy_maps": [[[0, 2]]],
}

TOOLS = (
    *STANDARD_TOOLS,
    *SKELETON_TOOLS,
    *CHAIN_TOOLS,
    *COMPLEX_CONVERSION_TOOLS,
    *PRODUCT_TOOLS,
    *COPRODUCT_TOOLS,
    *DEGENERACY_TOOLS,
    *IMAGE_TOOLS,
    *MAP_TOOLS,
    *TRUNCATE_TOOLS,
    *SUBSET_TOOLS,
    MathTool(
        operation_id="topology.simplicial_set.from_tables.compute",
        title="Check finite simplicial-set tables against every simplicial identity",
        description=(
            "Exhaust every in-range face/degeneracy identity for complete finite "
            "degree tables X_0..X_N. Return the canonical truncated simplicial "
            "set with its identity count, or the first unequal identity row. "
            "Degree levels may be empty; the all-empty tables represent the "
            "initial simplicial set. "
            "Tables must be complete: every degree carries all its simplices "
            "and every in-range face and degeneracy map as index rows."
        ),
        request_type=SimplicialSetTablesRequest,
        result_type=SimplicialSetTablesResult,
        run=_run_from_tables,
        tags=("topology", "simplicial-set", "face-map", "degeneracy-map", "exact"),
        discovery_terms=(
            "simplicial set",
            "face map",
            "degeneracy map",
            "simplicial identity",
            "truncated simplicial set",
        ),
        examples=(
            OperationExample(
                name="delta_one_prefix",
                description=(
                    "Check the degree-0..1 prefix of the 1-simplex Delta[1]; "
                    "face rows delete one entry and the degeneracy row "
                    "duplicates the entry."
                ),
                input=_DELTA_ONE_PREFIX,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
