# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.edge_paths.presentation_maps import *


def _run(r: Any) -> Any:
    return homomorphism(r.source, r.target, r.generator_images)


_P = {"generators": ["x"], "relators": []}
TOOLS = (
    MathTool(
        operation_id="topology.group_presentation.homomorphism.compute",
        title="Check a bounded induced presentation homomorphism",
        description="Map each source generator to a freely reduced target word and run a bounded normal-closure witness search for every source relator image; a missed witness is UNKNOWN rather than a mathematical negative.",
        request_type=PresentationHomomorphismRequest,
        result_type=PresentationHomomorphismResult,
        run=_run,
        tags=("topology", "presentation", "homomorphism", "exact"),
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
)
__all__ = ["TOOLS"]
