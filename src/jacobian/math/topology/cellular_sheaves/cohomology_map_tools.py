"""Catalog declaration for induced cellular-sheaf cohomology maps."""

from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.topology.cellular_sheaves.cohomology_maps import (
    SheafCohomologyMapRequest,
    SheafCohomologyMapResult,
    cohomology_map,
)


def _run(request: SheafCohomologyMapRequest) -> SheafCohomologyMapResult:
    return cohomology_map(request.morphism)


TOOLS: MathTools = (
    MathTool(
        operation_id="cellular_sheaf.morphism.cohomology_map.compute",
        title="Induce the map on cellular sheaf cohomology",
        description=(
            "Re-establish a natural cellular-sheaf morphism, compute the exact "
            "source and target cohomology groups, and return the induced linear "
            "map in their retained cocycle-representative bases."
        ),
        request_type=SheafCohomologyMapRequest,
        result_type=SheafCohomologyMapResult,
        run=_run,
        tags=("topology", "cellular-sheaf", "morphism", "cohomology", "exact"),
        discovery_terms=(
            "induced cellular sheaf cohomology map",
            "sheaf cohomology functoriality",
            "map on cellular sheaf cohomology",
        ),
    ),
)


__all__ = ["TOOLS"]
