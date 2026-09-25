"""Public operation for finite simplicial subobject prefixes."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets.standard import standard_simplex
from jacobian.math.topology.simplicial_sets.subset import simplicial_subset
from jacobian.math.topology.simplicial_sets.subset_models import (
    SimplicialSubsetPrefix,
    SimplicialSubsetRequest,
)

_DELTA_ONE = standard_simplex(1, 2).model_dump(mode="json")

TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.subset.from_degree_families.compute",
        title="Construct a finite simplicial subset prefix",
        description=(
            "Select a source-index family in every degree of one complete "
            "finite simplicial-set prefix. The operation checks closure under "
            "every visible face and degeneracy, then returns the reindexed "
            "subobject prefix and its exact injective inclusion map into the "
            "retained ambient prefix. Empty degree families are supported. "
            "The prefix ends at the source maximum degree and claims no maps "
            "above it; family cardinality, map incidence, work, and output "
            "bytes are admitted before construction."
        ),
        request_type=SimplicialSubsetRequest,
        result_type=SimplicialSubsetPrefix,
        run=simplicial_subset,
        tags=("topology", "simplicial-set", "subobject", "face", "degeneracy", "exact"),
        discovery_terms=(
            "simplicial subset",
            "simplicial subobject",
            "degreewise closed family",
            "subfunctor of simplicial set",
            "simplicial inclusion map",
        ),
        examples=(
            OperationExample(
                name="constant_vertex_subobject",
                description=(
                    "Select one vertex and its degeneracies in a Delta[1] "
                    "prefix; every selected simplex must remain in the "
                    "family under all visible faces and degeneracies."
                ),
                input={
                    "simplicial_set": _DELTA_ONE,
                    "degree_indices": [[0], [0], [0]],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
