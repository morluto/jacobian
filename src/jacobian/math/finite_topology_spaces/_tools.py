"""Finite topological space operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_topology_spaces._models import (
    BoundaryResult,
    ClosureResult,
    ContinuousCheckRequest,
    ContinuousCheckResult,
    InteriorResult,
    KolmogorovQuotientRequest,
    KolmogorovQuotientResult,
    SubsetRequest,
)
from jacobian.math.finite_topology_spaces._operations import (
    compute_boundary,
    compute_closure,
    compute_continuous_check,
    compute_interior,
    compute_kolmogorov_quotient,
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


# A Sierpinski space: points {a, b}, preorder rows: a -> {a}, b -> {a, b}
# (a <= b in specialization order, so open sets are {}, {a}, {a,b}).
_SPACE = {
    "points": ["a", "b"],
    "preorder": [[0], [0, 1]],
}


TOPOLOGY_SPACE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "topology.finite.interior.compute",
        "Compute the interior of a subset",
        "Return the largest open set contained in the subset. In an "
        "Alexandrov space, the interior consists of all points whose minimal "
        "open neighbourhood is contained in the subset.",
        SubsetRequest,
        InteriorResult,
        compute_interior,
        "finite-topology",
        "interior",
        "exact",
        examples=(
            example(
                "sierpinski_interior",
                "Interior of {b} in the Sierpinski space.",
                {"space": _SPACE, "subset": [1]},
            ),
        ),
    ),
    _op(
        "topology.finite.closure.compute",
        "Compute the closure of a subset",
        "Return the smallest closed set containing the subset. The closure "
        "of x is the up-set of x in the specialization preorder.",
        SubsetRequest,
        ClosureResult,
        compute_closure,
        "finite-topology",
        "closure",
        "exact",
        examples=(
            example(
                "sierpinski_closure",
                "Closure of {a} in the Sierpinski space.",
                {"space": _SPACE, "subset": [0]},
            ),
        ),
    ),
    _op(
        "topology.finite.boundary.compute",
        "Compute the boundary of a subset",
        "Return the boundary of a subset: closure minus interior.",
        SubsetRequest,
        BoundaryResult,
        compute_boundary,
        "finite-topology",
        "boundary",
        "exact",
        examples=(
            example(
                "sierpinski_boundary",
                "Boundary of {a} in the Sierpinski space.",
                {"space": _SPACE, "subset": [0]},
            ),
        ),
    ),
    _op(
        "topology.finite.kolmogorov_quotient.compute",
        "Compute the T0 (Kolmogorov) quotient",
        "Return the T0 quotient that identifies points with the same minimal "
        "open neighbourhood, plus the class map.",
        KolmogorovQuotientRequest,
        KolmogorovQuotientResult,
        compute_kolmogorov_quotient,
        "finite-topology",
        "kolmogorov-quotient",
        "exact",
        examples=(
            example(
                "sierpinski_kolmogorov",
                "T0 quotient of the Sierpinski space.",
                {"space": _SPACE},
            ),
        ),
    ),
    _op(
        "topology.finite.continuity_check.compute",
        "Check whether a point map is continuous",
        "Return whether a point map between finite topological spaces is "
        "continuous. A map f: X -> Y is continuous iff x' <= x implies "
        "f(x') <= f(x) in the specialization preorders.",
        ContinuousCheckRequest,
        ContinuousCheckResult,
        compute_continuous_check,
        "finite-topology",
        "continuity",
        "exact",
        examples=(
            example(
                "identity_continuous",
                "The identity map is continuous.",
                {
                    "point_map": {
                        "source": _SPACE,
                        "target": _SPACE,
                        "point_map": [0, 1],
                    },
                },
            ),
        ),
    ),
)

TOOLS = TOPOLOGY_SPACE_OPERATIONS

__all__ = ["TOOLS"]
