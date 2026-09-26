"""Published operation declarations for rational matrix Lie spans."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lie_algebras.matrix_span._models import (
    LieMatrixSpanRealization,
    LieMatrixSpanRequest,
)
from jacobian.math.lie_algebras.matrix_span.operations import (
    lie_algebra_from_matrix_span,
)


def _run(request: LieMatrixSpanRequest) -> LieMatrixSpanRealization:
    return lie_algebra_from_matrix_span(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="lie_algebra.from_matrix_span.compute",
        title="Construct a Lie algebra from a closed rational matrix span",
        description=(
            "Accept an ordered, linearly independent family of square rational "
            "matrices whose span is already closed under commutator. Return the "
            "induced finite-dimensional Lie algebra over QQ and retain the exact "
            "matrix basis in matching order. Reject dependent or nonclosed input; "
            "this operation does not generate a larger subalgebra. Matrix order "
            "and span dimension are at most 8, with admitted exact work and output."
        ),
        request_type=LieMatrixSpanRequest,
        result_type=LieMatrixSpanRealization,
        run=_run,
        tags=("lie-algebra", "matrix-span", "commutator", "exact", "rational"),
        discovery_terms=(
            "construct Lie algebra from matrix span",
            "matrix Lie algebra from closed subspace",
            "induced bracket on rational matrices",
            "commutator closed matrix basis",
        ),
        examples=(
            OperationExample(
                name="sl2_matrices",
                description="Construct the Lie algebra on the standard rational sl2 matrix basis.",
                input={"matrices": [
                    {"domain": "QQ", "entries": [[{"num": "0", "den": "1"}, {"num": "1", "den": "1"}], [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}]]},
                    {"domain": "QQ", "entries": [[{"num": "0", "den": "1"}, {"num": "0", "den": "1"}], [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}]]},
                    {"domain": "QQ", "entries": [[{"num": "1", "den": "1"}, {"num": "0", "den": "1"}], [{"num": "0", "den": "1"}, {"num": "-1", "den": "1"}]]},
                ]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
