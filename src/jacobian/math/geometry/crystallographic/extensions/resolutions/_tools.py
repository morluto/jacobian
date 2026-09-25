"""Public tool manifest for Bieberbach polygon free resolutions."""

from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.geometry.crystallographic.extensions.resolutions._models import (
    BieberbachPolygonFreeResolution,
    BieberbachPolygonFreeResolutionRequest,
)
from jacobian.math.geometry.crystallographic.extensions.resolutions.operations import (
    polygon_free_resolution,
)


def _compute(
    request: BieberbachPolygonFreeResolutionRequest,
) -> BieberbachPolygonFreeResolution:
    return polygon_free_resolution(request.source)


TOOLS: MathTools = (
    MathTool(
        operation_id="crystallographic.bieberbach.polygon_free_resolution.compute",
        title="Construct a Bieberbach polygon free resolution",
        description=(
            "Construct the finite cellular free ZGamma resolution through degree "
            "two from a verified torsion-free paired polygon. The exact sparse "
            "group-ring boundaries use the extension's lattice translations and "
            "holonomy indices; augmentation gives the integral quotient chains. "
            "The admitted scope is dimension two, at most 32 polygon vertices "
            "and facets, and the source-bound face-orbit contract."
        ),
        request_type=BieberbachPolygonFreeResolutionRequest,
        result_type=BieberbachPolygonFreeResolution,
        run=_compute,
        tags=("Bieberbach-group", "free-resolution", "group-homology", "exact"),
        discovery_terms=(
            "free integral group-ring resolution of a Bieberbach group",
            "cellular resolution from a flat polygon side pairing",
            "group homology resolution of a Klein bottle group",
        ),
        examples=(),
    ),
)

__all__ = ["TOOLS"]
