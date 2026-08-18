"""Finite topology operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_topology._models import (
    BeatPointsRequest,
    BeatPointsResult,
    ClosureRequest,
    ClosureResult,
    ConnectedComponentsRequest,
    ConnectedComponentsResult,
    InteriorRequest,
    InteriorResult,
    IsContinuousRequest,
    IsContinuousResult,
    SpecializationPreorderRequest,
    SpecializationPreorderResult,
)
from jacobian.math.finite_topology._operations import (
    compute_beat_points,
    compute_closure,
    compute_connected_components,
    compute_interior,
    compute_is_continuous,
    compute_specialization_preorder,
)


def _op[
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# Sierpinski topology: points {0, 1}, opens: {}, {1}, {0,1}
_TO3 = {
    "topology": {
        "point_count": 2,
        "open_sets": [
            [],
            [1],
            [0, 1],
        ],
    },
}

# Discrete topology on 3 points
_DISCRETE = {
    "topology": {
        "point_count": 3,
        "open_sets": [
            [],
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
            [0, 1, 2],
        ],
    },
}


FINITE_TOPOLOGY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "topology.specialization_preorder.compute",
        "Compute the specialization preorder of a finite topology",
        "Return the specialization preorder matrix of a finite topology. "
        "preorder[i][j] is True iff j is in the closure of {i}.",
        SpecializationPreorderRequest,
        SpecializationPreorderResult,
        compute_specialization_preorder,
        "topology",
        "preorder",
        "exact",
        examples=(
            example(
                "sierpinski_topology",
                "Specialization preorder of the Sierpinski topology.",
                _TO3,
            ),
        ),
    ),
    _op(
        "topology.closure.compute",
        "Compute the closure of a subset in a finite topology",
        "Return the smallest closed set containing the given subset.",
        ClosureRequest,
        ClosureResult,
        compute_closure,
        "topology",
        "closure",
        "exact",
        examples=(
            example(
                "sierpinski_closure",
                "Closure of {1} in the Sierpinski topology.",
                {"topology": _TO3["topology"], "subset": [1]},
            ),
        ),
    ),
    _op(
        "topology.interior.compute",
        "Compute the interior of a subset in a finite topology",
        "Return the largest open set contained in the given subset.",
        InteriorRequest,
        InteriorResult,
        compute_interior,
        "topology",
        "interior",
        "exact",
        examples=(
            example(
                "sierpinski_interior",
                "Interior of {0} in the Sierpinski topology.",
                {"topology": _TO3["topology"], "subset": [0]},
            ),
        ),
    ),
    _op(
        "topology.connected_components.compute",
        "Compute connected components of a finite topology",
        "Return the connected components as a partition of the point set, "
        "using the specialization preorder relation in both directions.",
        ConnectedComponentsRequest,
        ConnectedComponentsResult,
        compute_connected_components,
        "topology",
        "connected-components",
        "exact",
        examples=(
            example(
                "discrete_three_points",
                "Connected components of a discrete 3-point space.",
                _DISCRETE,
            ),
        ),
    ),
    _op(
        "topology.is_continuous.compute",
        "Check if a point map between topologies is continuous",
        "Return whether a point map f: X -> Y is continuous, i.e., the "
        "preimage of every open set in Y is open in X.",
        IsContinuousRequest,
        IsContinuousResult,
        compute_is_continuous,
        "topology",
        "continuous-map",
        "exact",
    ),
    _op(
        "topology.beat_points.compute",
        "Find all beat points of a finite topology",
        "Return the up and down beat points of a finite topology. A down beat "
        "point has a unique maximal element among its lower covers; an up beat "
        "point has a unique minimal element among its upper covers.",
        BeatPointsRequest,
        BeatPointsResult,
        compute_beat_points,
        "topology",
        "beat-points",
        "exact",
    ),
)

TOOLS = FINITE_TOPOLOGY_OPERATIONS

__all__ = ["TOOLS"]
