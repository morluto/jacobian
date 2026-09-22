# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.edge_paths.presentation_maps import *


def _run(r: Any) -> Any:
    return direct_relator_match(r.source, r.target, r.generator_images)


_P = {"generators": ["x"], "relators": []}
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
)
__all__ = ["TOOLS"]
