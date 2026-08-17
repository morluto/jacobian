"""Exact symbolic matrix operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.symbolic._models import (
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.matrices.symbolic._operations import (
    compute_symbolic_characteristic_polynomial,
    compute_symbolic_determinant,
    compute_symbolic_eigenvalues,
    compute_symbolic_rank,
)


def symbolic_matrix_operation[
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
                "Compute the determinant of [[a, c], [b, d]]; the matrix must be square and rectangular over declared variables.",
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
                "Compute the rank of a 2x2 symbolic matrix; rows must be nonempty and equal length over declared variables.",
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
        "matrix.symbolic.characteristic_polynomial.compute",
        "Compute an exact symbolic characteristic polynomial",
        "Compute the dense monic coefficients of det(lambda I - A) for a square symbolic matrix whose entries are rational functions in declared algebraically independent variables.",
        SymbolicMatrixRequest,
        SymbolicCharacteristicPolynomialResult,
        compute_symbolic_characteristic_polynomial,
        "matrix",
        "symbolic",
        "characteristic-polynomial",
        "rational-function-field",
        "exact",
        examples=(
            example(
                "symbolic_charpoly_two_by_two",
                "Compute the characteristic polynomial of [[a, c], [b, d]]; the matrix must be square and rectangular.",
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
        "matrix.symbolic.eigenvalues.compute",
        "Compute exact symbolic eigenvalues",
        "Compute the exact eigenvalues with algebraic multiplicities of a square symbolic matrix using SymPy's eigenvals. Entries may be rational functions in declared algebraically independent variables; eigenvalues are returned as canonical SymPy expression strings.",
        SymbolicMatrixRequest,
        SymbolicEigenvaluesResult,
        compute_symbolic_eigenvalues,
        "matrix",
        "symbolic",
        "eigenvalues",
        "rational-function-field",
        "exact",
        examples=(
            example(
                "symbolic_eigenvalues_two_by_two",
                "Compute the exact eigenvalues of [[1, 2], [3, 4]]; the matrix must be square and rectangular.",
                {
                    "matrix": {
                        "variables": [],
                        "entries": [["1", "2"], ["3", "4"]],
                    }
                },
            ),
        ),
    ),
)

TOOLS = SYMBOLIC_MATRIX_OPERATIONS

__all__ = ["TOOLS"]
