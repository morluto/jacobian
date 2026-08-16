"""Exact symbolic matrix operation declarations."""

from collections.abc import Callable

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.contracts.symbolic_matrix import (
    SymbolicDeterminantResult,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.domains._examples import example
from jacobian.domains.symbolic_matrix.operations import (
    compute_symbolic_determinant,
    compute_symbolic_rank,
)
from jacobian.math_tools import MathTool


def symbolic_matrix_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
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


SYMBOLIC_MATRIX_OPERATIONS = (
    symbolic_matrix_operation(
        "matrix.symbolic.determinant.compute",
        "Compute an exact symbolic matrix determinant over QQ(t_1, ..., t_n)",
        "Compute the determinant of a square matrix whose entries are rational functions in declared algebraically independent variables, using SymPy's exact fraction-free Bareiss algorithm.",
        SymbolicMatrixRequest,
        SymbolicDeterminantResult,
        compute_symbolic_determinant,
        "matrix",
        "symbolic",
        "determinant",
        "rational-function-field",
        "exact",
        examples=(
            example(
                "symbolic_determinant_two_by_two",
                "Compute the determinant of [[a, c], [b, d]].",
                {
                    "matrix": {
                        "variables": ["a", "b", "c", "d"],
                        "entries": [["a", "c"], ["b", "d"]],
                    }
                },
            ),
        ),
    ),
    symbolic_matrix_operation(
        "matrix.symbolic.rank.compute",
        "Compute exact symbolic matrix rank over QQ(t_1, ..., t_n)",
        "Compute the rank and RREF pivot columns of a rectangular matrix whose entries are rational functions in declared algebraically independent variables, using SymPy's exact row reduction.",
        SymbolicMatrixRequest,
        SymbolicRankResult,
        compute_symbolic_rank,
        "matrix",
        "symbolic",
        "rank",
        "rational-function-field",
        "exact",
        examples=(
            example(
                "symbolic_rank_full",
                "Compute the rank of a 2x2 symbolic matrix.",
                {
                    "matrix": {
                        "variables": ["a", "b", "c", "d"],
                        "entries": [["a", "c"], ["b", "d"]],
                    }
                },
            ),
        ),
    ),
)
