"""Catalog operation for finite simplicial-map images."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.image import (
    SimplicialMapImageRequest,
    SimplicialMapImageResult,
    simplicial_map_image,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex

_SOURCE = standard_simplex(1, 2)
_TARGET = standard_simplex(0, 2)
_COLLAPSE = {
    "simplicial_map": {
        "source": _SOURCE.model_dump(mode="json"),
        "target": _TARGET.model_dump(mode="json"),
        "maps": [[0, 0], [0, 0, 0], [0, 0, 0, 0]],
    }
}

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.map.image.compute",
        title="Factor a finite simplicial map through its image",
        description=(
            "Compute the degreewise image sub-simplicial-set of a finite-prefix "
            "simplicial map, returning the exact surjection onto that image and "
            "its inclusion into the target. The operation rechecks both source "
            "and target simplicial identities and every visible naturality square."
        ),
        request_type=SimplicialMapImageRequest,
        result_type=SimplicialMapImageResult,
        run=simplicial_map_image,
        tags=("topology", "simplicial-set", "map", "image", "exact"),
        discovery_terms=(
            "simplicial map image",
            "image subobject",
            "image factorization",
        ),
        examples=(
            OperationExample(
                name="collapse_delta_one",
                description=(
                    "Factor the constant map from the 1-simplex to the terminal "
                    "simplicial set through its degreewise image."
                ),
                input=_COLLAPSE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
