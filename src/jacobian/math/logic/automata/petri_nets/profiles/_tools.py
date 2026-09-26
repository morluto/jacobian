"""Petri-net reachability-profile operation declaration."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.petri_nets.profiles._models import (
    ReachabilityTokenProfileRequest,
    ReachabilityTokenProfileResult,
)
from jacobian.math.logic.automata.petri_nets.profiles.operations import (
    reachability_token_profile,
)


def compute_reachability_token_profile(
    request: ReachabilityTokenProfileRequest,
) -> ReachabilityTokenProfileResult:
    return reachability_token_profile(request.source_graph)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="petri_net.reachability.token_profile.compute",
        title="Profile token ranges in a Petri-net reachability graph",
        description=(
            "Compute exact per-place and total-token extrema over represented "
            "reachable markings. The profile covers the full reachable set only "
            "when the source graph is complete; truncation yields observed-prefix extrema."
        ),
        request_type=ReachabilityTokenProfileRequest,
        result_type=ReachabilityTokenProfileResult,
        run=compute_reachability_token_profile,
        tags=("petri-net", "reachability", "token-profile", "exact"),
        discovery_terms=(
            "Petri-net reachable token minimum maximum",
            "place boundedness from complete finite reachability graph",
            "token extrema per place",
        ),
        examples=(
            OperationExample(
                name="two_marking_bounded_place",
                description=(
                    "A marked place that drains to empty reports both extrema."
                ),
                input={
                    "source_graph": {
                        "net": {
                            "place_count": 1,
                            "transition_count": 1,
                            "pre": [[1]],
                            "post": [[0]],
                        },
                        "initial_marking": {"tokens": [1]},
                        "max_states": 2,
                        "states": [
                            {
                                "state_index": 0,
                                "place_axis": [0],
                                "marking": {"tokens": [1]},
                            },
                            {
                                "state_index": 1,
                                "place_axis": [0],
                                "marking": {"tokens": [0]},
                            },
                        ],
                        "edges": [
                            {
                                "source_state": 0,
                                "transition": 0,
                                "target_state": 1,
                            }
                        ],
                        "truncated": False,
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
