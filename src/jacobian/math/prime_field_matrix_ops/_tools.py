"""Prime-field matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.prime_field_matrix_ops._models import (
    NullspaceRequest,
    NullspaceResult,
    RankRequest,
    RankResult,
    RrefRequest,
    RrefResult,
)
from jacobian.math.prime_field_matrix_ops._operations import (
    compute_nullspace,
    compute_rank,
    compute_rref,
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
    examples: tuple[OperationExample, ...],
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_MATRIX = {
    "matrix": {
        "prime": 2,
        "entries": [[1, 1, 0], [0, 1, 1]],
        "columns": 3,
    }
}

_TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "prime_field_matrix.rank.compute",
        "Compute matrix rank over GF(p)",
        "Compute the exact rank of a bounded integer matrix over an explicit "
        "prime field using DomainMatrix Gaussian elimination.",
        RankRequest,
        RankResult,
        compute_rank,
        "linear-algebra",
        "finite-field",
        "rank",
        "exact",
        examples=(
            example(
                "rank_over_gf2",
                "Compute the rank of a 2x3 matrix over GF(2).",
                _MATRIX,
            ),
        ),
    ),
    _op(
        "prime_field_matrix.rref.compute",
        "Compute reduced row-echelon form over GF(p)",
        "Compute the exact reduced row-echelon form and pivot columns of a "
        "bounded integer matrix over an explicit prime field.",
        RrefRequest,
        RrefResult,
        compute_rref,
        "linear-algebra",
        "finite-field",
        "rref",
        "exact",
        examples=(
            example(
                "rref_over_gf2",
                "Compute the RREF of a 2x3 matrix over GF(2).",
                _MATRIX,
            ),
        ),
    ),
    _op(
        "prime_field_matrix.nullspace.compute",
        "Compute nullspace over GF(p)",
        "Compute a deterministic basis of the right nullspace of a bounded "
        "integer matrix over an explicit prime field.",
        NullspaceRequest,
        NullspaceResult,
        compute_nullspace,
        "linear-algebra",
        "finite-field",
        "nullspace",
        "exact",
        examples=(
            example(
                "nullspace_over_gf2",
                "Compute the nullspace of a 2x3 matrix over GF(2).",
                _MATRIX,
            ),
        ),
    ),
)

TOOLS = _TOOLS

__all__ = ["TOOLS"]
