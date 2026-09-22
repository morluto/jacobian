# ruff: noqa: F403,F405
from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.discrete_morse.extensions import *


def _greedy(r: Any) -> Any:
    return greedy_matching(r)


def _collapse(r: Any) -> Any:
    return collapse_sequence(r)


_I = {"vertices": ["a", "b"], "facets": [["a", "b"]]}
TOOLS = (
    MathTool(
        operation_id="topology.discrete_morse.greedy_matching.compute",
        title="Construct a deterministic greedy Morse matching",
        description="Match the first available codimension-one face/coface pair in canonical face order and classify the resulting matching exactly; this is not a minimum matching claim.",
        request_type=GreedyMatchingRequest,
        result_type=__import__(
            "jacobian.math.topology.discrete_morse._models",
            fromlist=["DiscreteMorseMatchingResult"],
        ).DiscreteMorseMatchingResult,
        run=_greedy,
        tags=("topology", "discrete-morse", "greedy", "exact"),
        examples=(
            OperationExample(
                name="interval_greedy",
                description="Construct the deterministic greedy matching of one interval; face/coface order is canonical.",
                input={"complex": _I},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_complex.collapse_sequence.compute",
        title="Apply a supplied elementary-collapse sequence",
        description="Apply each supplied free-face collapse to a source-bound simplicial complex, rejecting a non-free step and returning the exact residual complex.",
        request_type=CollapseSequenceRequest,
        result_type=CollapseSequenceResult,
        run=_collapse,
        tags=("topology", "simplicial", "collapse", "exact"),
        examples=(
            OperationExample(
                name="interval_collapse",
                description="Collapse the interval by removing vertex a together with edge ab; the pair must be a free codimension-one inclusion.",
                input={"complex": _I, "pairs": [{"face": ["a"], "coface": ["a", "b"]}]},
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
