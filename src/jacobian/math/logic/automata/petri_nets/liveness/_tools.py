"""Petri-net transition-liveness operation declaration."""

from typing import Any

from jacobian.catalog.models import MathTool
from jacobian.math.logic.automata.petri_nets.liveness._models import (
    TransitionLivenessRequest,
    TransitionLivenessResult,
)
from jacobian.math.logic.automata.petri_nets.liveness.operations import (
    transition_liveness_profile,
)


def compute_transition_liveness(
    request: TransitionLivenessRequest,
) -> TransitionLivenessResult:
    return transition_liveness_profile(request.source_graph)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="petri_net.transition_liveness.profile.compute",
        title="Classify Petri-net transition liveness",
        description=(
            "For each transition, determine whether it can eventually fire "
            "from every marking reachable in the supplied complete finite "
            "reachability graph. A truncated graph yields UNKNOWN for every "
            "transition; firing once somewhere does not establish liveness."
        ),
        request_type=TransitionLivenessRequest,
        result_type=TransitionLivenessResult,
        run=compute_transition_liveness,
        tags=("petri-net", "reachability", "liveness", "exact"),
        discovery_terms=(
            "Petri-net transition liveness",
            "L4 liveness",
            "transition can fire from every reachable marking",
        ),
        examples=(),
    ),
)

__all__ = ["TOOLS"]
