"""Eventual hitting profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.probability.markov_chains.eventual_hitting._models import (
    EventualHittingProfileRequest,
    EventualHittingProfileResult,
)
from jacobian.math.probability.markov_chains.eventual_hitting.operations import (
    compute_eventual_hitting_profile,
)
from jacobian.math.probability.markov_chains.values import as_transition_matrix


def compute_ehp_op(
    request: EventualHittingProfileRequest,
) -> EventualHittingProfileResult:
    return compute_eventual_hitting_profile(
        as_transition_matrix(request.matrix),
        request.target_states,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="probability.markov_chain.eventual_hitting_profile.compute",
        title="Compute the eventual hitting probability profile of a Markov chain",
        description=(
            "For one bounded exact finite Markov chain and one nonempty target "
            "state set, return the complete exact vector of probabilities that "
            "the target is ever hit from each source state."
        ),
        request_type=EventualHittingProfileRequest,
        result_type=EventualHittingProfileResult,
        run=compute_ehp_op,
        tags=("probability", "exact"),
        examples=(
            OperationExample(
                name="two_state",
                description="Two-state chain with target = absorbing state 1.",
                input={
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                    "target_states": [1],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
