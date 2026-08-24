"""Linear matroid operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.matroids._models import (
    MatroidClosureRequest,
    MatroidClosureResult,
)
from jacobian.math.matroids._operations import (
    _closure_invariant,
)


def compute_closure(request: MatroidClosureRequest) -> MatroidClosureResult:
    """Compute the closure (smallest flat) of a subset in a linear matroid."""
    closure, subset_rank = _closure_invariant(request.matroid, list(request.subset))
    return MatroidClosureResult(
        matroid=request.matroid,
        subset=request.subset,
        closure=closure,
        rank=subset_rank,
    )


_CLOSURE_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "matrix": {
            "prime": 5,
            "entries": [[1, 0, 1], [0, 1, 1]],
            "columns": 3,
        },
    },
    "subset": [0, 1],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matroid.closure.compute",
        version="1",
        title="Compute the closure of a subset in a linear matroid",
        description=(
            "Compute the closure (smallest flat) of a subset S in a matroid "
            "represented by the columns of a canonical matrix over GF(p). "
            "The closure adds all elements that lie in the span of S."
        ),
        request_type=MatroidClosureRequest,
        result_type=MatroidClosureResult,
        run=compute_closure,
        tags=("matroid", "closure", "flat", "exact"),
        examples=(
            example(
                "closure_of_basis",
                "Compute the closure of {0, 1} in a rank-2 matroid.",
                _CLOSURE_EXAMPLE,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
