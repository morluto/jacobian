"""Affine-map word collision profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.algebra.affine_map_word_collision._models import (
    WordCollisionProfileRequest,
    WordCollisionProfileResult,
)
from jacobian.math.algebra.affine_map_word_collision.operations import (
    compute_word_collision_profile,
)


def compute_word_collision_profile_op(
    request: WordCollisionProfileRequest,
) -> WordCollisionProfileResult:

    generators = tuple(
        (g.slope.as_fraction(), g.intercept.as_fraction()) for g in request.generators
    )
    return compute_word_collision_profile(
        generators, request.depth, enforce_transport=True
    )


def awc_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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
    awc_operation(
        "algebra.affine_map.word_collision_profile.compute",
        "Compute the word collision profile of an affine-map family",
        (
            "Given a finite indexed family of exact univariate affine maps "
            "x -> a_i*x + b_i and a fixed positive word length d, return the "
            "complete partition of all generator words of length d by their "
            "exact composed affine map. Convention: word (i_1,...,i_d) "
            "represents f_{i_d} o ... o f_{i_1}."
        ),
        WordCollisionProfileRequest,
        WordCollisionProfileResult,
        compute_word_collision_profile_op,
        "algebra",
        "affine-maps",
        "exact",
        examples=(
            example(
                "two_identity_maps",
                "Two copies of x->x+1 at depth 1 collide into one class.",
                {
                    "generators": [
                        {
                            "slope": {"num": "1", "den": "1"},
                            "intercept": {"num": "1", "den": "1"},
                        },
                        {
                            "slope": {"num": "1", "den": "1"},
                            "intercept": {"num": "1", "den": "1"},
                        },
                    ],
                    "depth": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
