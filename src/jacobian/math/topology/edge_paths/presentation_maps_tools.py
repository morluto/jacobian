# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.edge_paths.presentation_maps import *


def _run(r: Any) -> Any:
    return direct_relator_match(r.source, r.target, r.generator_images)


def _run_induced(r: Any) -> Any:
    return induced_fundamental_group_map(r)


_P = {"generators": ["x"], "relators": []}
_CIRCLE = canonical_complex(
    ("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))
).model_dump(mode="json")
TOOLS = (
    MathTool(
        operation_id="topology.group_presentation.direct_relator_match.compute",
        title="Check direct relator-match compatibility",
        description="Return a complete finite witness for the sufficient condition that every source relator maps literally to a target relator or its inverse. Absence of this witness does not decide normal-closure membership or deny a presentation homomorphism.",
        request_type=DirectRelatorMatchRequest,
        result_type=DirectRelatorMatchResult,
        run=_run,
        tags=("topology", "presentation", "relator-match", "exact"),
        examples=(
            OperationExample(
                name="free_generator_identity",
                description="Map the one-generator free presentation to itself; the presentation has no relators to preserve.",
                input={
                    "source": _P,
                    "target": _P,
                    "generator_images": [
                        {"letters": [{"generator": 0, "exponent": 1}]}
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial.fundamental_group.induced_map.compute",
        title="Construct the induced fundamental-group presentation map",
        description=(
            "Given an exact simplicial vertex map and compatible source/target "
            "basepoints, derive each source generator image using the canonical "
            "spanning trees, together with its exact generator-axis matrix on "
            "abelianizations. Return a target-relator conjugacy witness for every "
            "source triangle relation; degenerate triangle images freely reduce "
            "to the identity."
        ),
        request_type=FundamentalGroupMapRequest,
        result_type=FundamentalGroupMapResult,
        run=_run_induced,
        tags=("topology", "fundamental-group", "simplicial-map", "exact"),
        discovery_terms=(
            "induced fundamental group map",
            "simplicial map on pi1 presentations",
            "generator word images",
        ),
        examples=(
            OperationExample(
                name="circle_identity_fundamental_group_map",
                description=(
                    "The identity simplicial map of a 3-edge circle sends its "
                    "canonical fundamental-group generator to itself."
                ),
                input={
                    "map": {
                        "source": _CIRCLE,
                        "target": _CIRCLE,
                        "vertex_map": ["a", "b", "c"],
                    },
                    "source_base_vertex": "a",
                    "target_base_vertex": "a",
                },
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
