"""Eventual hitting profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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
        enforce_transport_limit=True,
    )


def ehp_action[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: MathTools = (
    ehp_action(
        "probability.markov_chain.eventual_hitting_profile.compute",
        "Compute the eventual hitting probability profile of a Markov chain",
        (
            "For one bounded exact finite Markov chain and one nonempty target "
            "state set, return the complete exact vector of probabilities that "
            "the target is ever hit from each source state."
        ),
        EventualHittingProfileRequest,
        EventualHittingProfileResult,
        compute_ehp_op,
        "probability",
        "exact",
        examples=(
            example(
                "two_state",
                "Two-state chain with target = absorbing state 1.",
                {
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
