"""Public declaration for finite truncated simplicial sets."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.chain_complexes.values import ChainMapValue
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
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialHomologyMapValue,
    TruncatedSimplicialMap,
    induced_normalized_chain_map,
    induced_normalized_homology_map,
)
from jacobian.math.topology.simplicial_sets.maps_tools import TOOLS as MAP_TOOLS
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.product_tools import TOOLS as PRODUCT_TOOLS
from jacobian.math.topology.simplicial_sets.quotient_tools import (
    TOOLS as QUOTIENT_TOOLS,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex
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


def _run_induced_chain_map(
    request: TruncatedSimplicialMap,
) -> ChainMapValue:
    return induced_normalized_chain_map(request)


def _run_induced_homology_map(
    request: TruncatedSimplicialMap,
) -> SimplicialHomologyMapValue:
    return induced_normalized_homology_map(request)


# Delta[1] truncated to degrees 0..1: X_0 = {0, 1}, X_1 = {00, 01, 11};
# faces delete one entry and s_0 duplicates the entry.
_DELTA_ONE_PREFIX = {
    "max_degree": 1,
    "sets": [["0", "1"], ["00", "01", "11"]],
    "face_maps": [[[0, 1, 1], [0, 0, 1]]],
    "degeneracy_maps": [[[0, 2]]],
}
_DELTA_ONE = standard_simplex(1, 2).model_dump(mode="json")
_IDENTITY_DELTA_ONE_MAP = {
    "source": _DELTA_ONE,
    "target": _DELTA_ONE,
    "maps": [[0, 1], [0, 1, 2], [0, 1, 2, 3]],
}

TOOLS = (
    *STANDARD_TOOLS,
    *CHAIN_TOOLS,
    *COMPLEX_CONVERSION_TOOLS,
    *PRODUCT_TOOLS,
    *COPRODUCT_TOOLS,
    *DEGENERACY_TOOLS,
    *IMAGE_TOOLS,
    *MAP_TOOLS,
    *TRUNCATE_TOOLS,
    *QUOTIENT_TOOLS,
    *SUBSET_TOOLS,
    MathTool(
        operation_id="topology.simplicial_set.map.induced_chain_map.compute",
        title="Compute the induced map on normalized chains",
        description=(
            "For a finite simplicial map, return its degreewise map on the "
            "normalized chain complexes with the same source and target "
            "basis labels. Recheck simplicial naturality and the chain-map "
            "identity. Degree is bounded by 4; admission bounds work and "
            "exact output size before matrix construction."
        ),
        request_type=TruncatedSimplicialMap,
        result_type=ChainMapValue,
        run=_run_induced_chain_map,
        tags=("topology", "simplicial-set", "map", "normalized-chains", "exact"),
        discovery_terms=(
            "induced normalized chain map",
            "simplicial map on chains",
            "simplicial homology map",
        ),
        examples=(
            OperationExample(
                name="identity_delta_one",
                description="Take the normalized chain map induced by the identity on Delta[1].",
                input=_IDENTITY_DELTA_ONE_MAP,
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_set.map.induced_homology.compute",
        title="Compute the induced map on supported normalized homology",
        description=(
            "Return the exact induced maps in degrees 0 through N-1 for a "
            "finite simplicial map on prefixes through degree N. Endpoints "
            "retain normalized chain representatives and canonical free and "
            "torsion generators. The top homology is omitted because its "
            "incoming differential is outside the prefix. Work and output "
            "are bounded by the normalized integral homology and map limits."
        ),
        request_type=TruncatedSimplicialMap,
        result_type=SimplicialHomologyMapValue,
        run=_run_induced_homology_map,
        tags=("topology", "simplicial-set", "map", "homology", "exact"),
        discovery_terms=(
            "induced simplicial homology map",
            "map on normalized homology",
            "simplicial map induced homology",
        ),
        examples=(
            OperationExample(
                name="identity_delta_one",
                description=(
                    "Compute the induced map on H0 and H1 for the identity "
                    "of Delta[1] with its degree-2 prefix."
                ),
                input=_IDENTITY_DELTA_ONE_MAP,
            ),
        ),
    ),
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
