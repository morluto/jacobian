"""Catalog operation for finite simplicial-map preimages."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.map_preimage import (
    SimplicialMapPreimageRequest,
    SimplicialMapPreimageResult,
    simplicial_map_preimage,
)
from jacobian.math.topology.simplicial_sets.maps import TruncatedSimplicialMap
from jacobian.math.topology.simplicial_sets.standard import standard_simplex
from jacobian.math.topology.simplicial_sets.subset import simplicial_subset
from jacobian.math.topology.simplicial_sets.subset_models import (
    SimplicialSubsetRequest,
)

_DELTA_ONE = standard_simplex(1, 1)
_DELTA_ONE_SUBSET = simplicial_subset(
    SimplicialSubsetRequest(simplicial_set=_DELTA_ONE, degree_indices=((0,), (0,)))
)
_IDENTITY = TruncatedSimplicialMap(
    source=_DELTA_ONE,
    target=_DELTA_ONE,
    maps=((0, 1), (0, 1, 2)),
)
_PREIMAGE_EXAMPLE = {
    "simplicial_map": _IDENTITY.model_dump(mode="json"),
    "target_subset": _DELTA_ONE_SUBSET.model_dump(mode="json"),
}

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.map.preimage.compute",
        title="Pull back a finite simplicial subobject",
        description=(
            "Compute the degreewise preimage of a complete finite-prefix "
            "simplicial subobject along a simplicial map. Return the source "
            "subobject and the restricted map into the selected target. "
            "The operation checks all visible carrier identities and both "
            "naturality relations, then admits work and result size before "
            "construction."
        ),
        request_type=SimplicialMapPreimageRequest,
        result_type=SimplicialMapPreimageResult,
        run=simplicial_map_preimage,
        tags=("topology", "simplicial-set", "map", "preimage", "exact"),
        discovery_terms=(
            "simplicial map preimage",
            "inverse image subobject",
            "pullback of simplicial subset",
        ),
        examples=(
            OperationExample(
                name="identity_preimage_delta_one_vertex",
                description=(
                    "Pull a vertex and its visible degeneracy back along the "
                    "identity map of a Delta[1] prefix."
                ),
                input=_PREIMAGE_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
