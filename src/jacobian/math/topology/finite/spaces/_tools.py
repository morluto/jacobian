"""Finite topological space operation declarations."""

from typing import Any

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


# A Sierpinski space: points {a, b}, preorder rows: a -> {a}, b -> {a, b}
# (a <= b in specialization order, so open sets are {}, {a}, {a,b}).
_SPACE = {
    "points": ["a", "b"],
    "preorder": [[0], [0, 1]],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.finite.interior.compute",
        title="Compute the interior of a subset",
        description="Return the largest open set contained in the subset. In an "
        "Alexandrov space, the interior consists of all points whose minimal "
        "open neighbourhood is contained in the subset.",
        request_type=SubsetRequest,
        result_type=InteriorResult,
        run=_interior,
        tags=("finite-topology", "interior", "exact"),
        examples=(
            OperationExample(
                name="sierpinski_interior",
                description="Interior of {b} in the Sierpinski space.",
                input={"space": _SPACE, "subset": [1]},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.finite.closure.compute",
        title="Compute the closure of a subset",
        description="Return the smallest closed set containing the subset. The closure "
        "of x is the up-set of x in the specialization preorder.",
        request_type=SubsetRequest,
        result_type=ClosureResult,
        run=_closure,
        tags=("finite-topology", "closure", "exact"),
        examples=(
            OperationExample(
                name="sierpinski_closure",
                description="Closure of {a} in the Sierpinski space.",
                input={"space": _SPACE, "subset": [0]},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.finite.boundary.compute",
        title="Compute the boundary of a subset",
        description="Return the boundary of a subset: closure minus interior.",
        request_type=SubsetRequest,
        result_type=BoundaryResult,
        run=_boundary,
        tags=("finite-topology", "boundary", "exact"),
        examples=(
            OperationExample(
                name="sierpinski_boundary",
                description="Boundary of {a} in the Sierpinski space.",
                input={"space": _SPACE, "subset": [0]},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.finite.kolmogorov_quotient.compute",
        title="Compute the T0 (Kolmogorov) quotient",
        description="Return the T0 quotient that identifies points with the same minimal "
        "open neighbourhood, plus the class map.",
        request_type=KolmogorovQuotientRequest,
        result_type=KolmogorovQuotientResult,
        run=_kolmogorov_quotient,
        tags=("finite-topology", "kolmogorov-quotient", "exact"),
        examples=(
            OperationExample(
                name="sierpinski_kolmogorov",
                description="T0 quotient of the Sierpinski space.",
                input={"space": _SPACE},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.finite.continuity_check.compute",
        title="Check whether a point map is continuous",
        description="Return whether a point map between finite topological spaces is "
        "continuous. A map f: X -> Y is continuous iff x' <= x implies "
        "f(x') <= f(x) in the specialization preorders.",
        request_type=ContinuousCheckRequest,
        result_type=ContinuousCheckResult,
        run=_continuous_check,
        tags=("finite-topology", "continuity", "exact"),
        examples=(
            OperationExample(
                name="identity_continuous",
                description="The identity map is continuous.",
                input={
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
