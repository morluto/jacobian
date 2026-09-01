"""Affine-map word collision profile operation declarations."""

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


TOOLS: MathTools = (
    MathTool(
        operation_id="algebra.affine_map.word_collision_profile.compute",
        title="Compute the word collision profile of an affine-map family",
        description=(
            "Given a finite indexed family of exact univariate affine maps "
            "x -> a_i*x + b_i and a fixed positive word length d, return the "
            "complete partition of all generator words of length d by their "
            "exact composed affine map. Convention: word (i_1,...,i_d) "
            "represents f_{i_d} o ... o f_{i_1}."
        ),
        request_type=WordCollisionProfileRequest,
        result_type=WordCollisionProfileResult,
        run=compute_word_collision_profile_op,
        tags=("algebra", "affine-maps", "exact"),
        examples=(
            OperationExample(
                name="two_identity_maps",
                description="Two copies of x->x+1 at depth 1 collide into one class.",
                input={
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
