"""Finite topological space operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.topology.finite.spaces._models import (
    BoundaryResult,
    ClosureResult,
    ContinuousCheckRequest,
    ContinuousCheckResult,
    InteriorResult,
    KolmogorovQuotientRequest,
    KolmogorovQuotientResult,
    SubsetRequest,
)
from jacobian.math.topology.finite.spaces.operations import (
    boundary,
    closure,
    continuous_check,
    interior,
    kolmogorov_quotient,
)


def _admit_subset(request: SubsetRequest) -> frozenset[int]:
    if any(not 0 <= index < len(request.space.points) for index in request.subset):
        raise OperationDomainValidationError(
            location=("subset",),
            code="finite_topology_space.subset_index_out_of_range",
            message="subset index out of range",
        )
    return frozenset(request.subset)


def _interior(request: SubsetRequest) -> InteriorResult:
    result = interior(request.space, _admit_subset(request))
    return InteriorResult(interior=tuple(sorted(result)))


def _closure(request: SubsetRequest) -> ClosureResult:
    result = closure(request.space, _admit_subset(request))
    return ClosureResult(closure=tuple(sorted(result)))


def _boundary(request: SubsetRequest) -> BoundaryResult:
    result = boundary(request.space, _admit_subset(request))
    return BoundaryResult(boundary=tuple(sorted(result)))


def _continuous_check(request: ContinuousCheckRequest) -> ContinuousCheckResult:
    result = continuous_check(request.point_map)
    return ContinuousCheckResult(is_continuous=result)


def _kolmogorov_quotient(
    request: KolmogorovQuotientRequest,
) -> KolmogorovQuotientResult:
    return kolmogorov_quotient(request.space)


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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "topology.finite.interior.compute",
        "Compute the interior of a subset",
        "Return the largest open set contained in the subset. In an "
        "Alexandrov space, the interior consists of all points whose minimal "
        "open neighbourhood is contained in the subset.",
        SubsetRequest,
        InteriorResult,
        _interior,
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
        _closure,
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
        _boundary,
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
        _kolmogorov_quotient,
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
        _continuous_check,
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


__all__ = ["TOOLS"]
