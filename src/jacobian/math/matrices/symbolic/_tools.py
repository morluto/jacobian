"""Exact symbolic matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from sympy.matrices.exceptions import MatrixError

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.symbolic._models import (
    SymbolicCharacteristicPolynomialRequest,
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantRequest,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicLinearSystemRequest,
    SymbolicLinearSystemResult,
    SymbolicMatrix,
    SymbolicMatrixProductRequest,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.matrices.symbolic.operations import (
    _symbolic_characteristic_polynomial_kernel,
    symbolic_characteristic_polynomial,
    symbolic_determinant,
    symbolic_eigenvalues,
    symbolic_linear_system_solve,
    symbolic_matrix_multiply,
    symbolic_rank,
)


def _run_determinant(request: SymbolicDeterminantRequest) -> SymbolicDeterminantResult:
    return SymbolicDeterminantResult(
        determinant=symbolic_determinant(
            request.matrix.entries,
            request.matrix.variables,
        )
    )


def _run_rank(request: SymbolicMatrixRequest) -> SymbolicRankResult:
    rank, pivot_columns = symbolic_rank(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicRankResult(rank=rank, pivot_columns=pivot_columns)


def _run_product(request: SymbolicMatrixProductRequest) -> SymbolicMatrix:
    return symbolic_matrix_multiply(request.left, request.right)


def _run_characteristic(
    request: SymbolicCharacteristicPolynomialRequest,
) -> SymbolicCharacteristicPolynomialResult:
    degree, coefficients = symbolic_characteristic_polynomial(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicCharacteristicPolynomialResult(
        degree=degree,
        coefficients_descending=coefficients,
    )


def _run_eigenvalues(
    request: SymbolicCharacteristicPolynomialRequest,
) -> SymbolicEigenvaluesResult:
    try:
        eigenvalues = symbolic_eigenvalues(
            request.matrix.entries,
            request.matrix.variables,
        )
    except MatrixError:
        degree, coefficients = _symbolic_characteristic_polynomial_kernel(
            request.matrix.entries,
            request.matrix.variables,
        )
        return SymbolicEigenvaluesResult(
            representation="ROOTS_BY_POLYNOMIAL",
            characteristic_polynomial=coefficients,
            degree=degree,
        )
    return SymbolicEigenvaluesResult(
        representation="EXPLICIT_ROOTS",
        eigenvalues=tuple(value for value, _ in eigenvalues),
        multiplicities=tuple(mult for _, mult in eigenvalues),
    )


def _run_linear_system(
    request: SymbolicLinearSystemRequest,
) -> SymbolicLinearSystemResult:
    classification, solution, particular, nullspace = symbolic_linear_system_solve(
        request.matrix.entries,
        request.rhs,
        request.matrix.variables,
    )
    return SymbolicLinearSystemResult._from_kernel(
        matrix=request.matrix,
        rhs=request.rhs,
        classification=classification,
        solution=solution,
        particular_solution=particular,
        nullspace_basis=nullspace,
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


def _rational_function(
    variables: tuple[str, ...],
    *numerator_terms: tuple[int, tuple[int, ...]],
) -> dict[str, Any]:
    def polynomial(
        terms: tuple[tuple[int, tuple[int, ...]], ...],
    ) -> dict[str, Any]:
        return {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": list(exponents),
                }
                for coefficient, exponents in sorted(
                    terms, key=lambda t: t[1], reverse=True
                )
                if coefficient != 0
            ]
        }

    return {
        "domain": "QQ",
        "variables": list(variables),
        "numerator": polynomial(numerator_terms),
        "denominator": polynomial(((1, (0,) * len(variables)),)),
    }


def _generic_two_by_two() -> dict[str, Any]:
    variables = ("a", "b", "c", "d")
    return {
        "variables": list(variables),
        "entries": [
            [
                _rational_function(variables, (1, (1, 0, 0, 0))),
                _rational_function(variables, (1, (0, 0, 1, 0))),
            ],
            [
                _rational_function(variables, (1, (0, 1, 0, 0))),
                _rational_function(variables, (1, (0, 0, 0, 1))),
            ],
        ],
    }


_LINEAR_SYSTEM_EXAMPLE = example(
    "symbolic_linear_system_unique",
    "Solve [[1, t], [0, 1]] x = [t, 1] over QQ(t); solution is x = [0, 1].",
    {
        "matrix": {
            "variables": ["t"],
            "entries": [
                [
                    _rational_function(("t",), (1, (0,))),
                    _rational_function(("t",), (1, (1,))),
                ],
                [
                    _rational_function(("t",), (0, (0,))),
                    _rational_function(("t",), (1, (0,))),
                ],
            ],
        },
        "rhs": [
            _rational_function(("t",), (1, (1,))),
            _rational_function(("t",), (1, (0,))),
        ],
    },
)


_SYMBOLIC_PRODUCT_EXAMPLE = example(
    "symbolic_matrix_product",
    "Multiply [[a, b]] by the column [[1], [1]] over QQ(a, b); both matrices must use the same ordered field and compatible inner dimension.",
    {
        "left": {
            "variables": ["a", "b"],
            "entries": [
                [
                    _rational_function(("a", "b"), (1, (1, 0))),
                    _rational_function(("a", "b"), (1, (0, 1))),
                ]
            ],
        },
        "right": {
            "variables": ["a", "b"],
            "entries": [
                [_rational_function(("a", "b"), (1, (0, 0)))],
                [_rational_function(("a", "b"), (1, (0, 0)))],
            ],
        },
    },
)


TOOLS = (
    symbolic_matrix_operation(
        "matrix.symbolic.determinant.compute",
        "Compute an exact symbolic matrix determinant (det) over QQ(t_1, ..., t_n)",
        "Compute the determinant of a square matrix whose entries are rational functions in declared algebraically independent variables, using SymPy's exact fraction-free Bareiss algorithm.",
        SymbolicDeterminantRequest,
        SymbolicDeterminantResult,
        _run_determinant,
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
                    "matrix": _generic_two_by_two(),
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
        _run_rank,
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
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
    ),
    symbolic_matrix_operation(
        "matrix.symbolic.multiply.compute",
        "Multiply exact symbolic matrices over QQ(t_1, ..., t_n)",
        (
            "Compute the row-by-column product of two compatible symbolic matrices "
            "over one explicitly ordered rational-function field. Every product "
            "entry is returned as a canonical reduced rational function; admission "
            "bounds unreduced expansion plus cancellation-safe canonical and "
            "aggregate support, exponents, coefficients, and result before SymPy "
            "multiplication."
        ),
        SymbolicMatrixProductRequest,
        SymbolicMatrix,
        _run_product,
        "matrix",
        "symbolic",
        "matrix-multiplication",
        "product",
        "rational-function-field",
        "exact",
        examples=(_SYMBOLIC_PRODUCT_EXAMPLE,),
    ),
    symbolic_matrix_operation(
        "matrix.symbolic.characteristic_polynomial.compute",
        "Compute an exact symbolic characteristic polynomial",
        "Compute the dense monic coefficients of det(lambda I - A) for a square symbolic matrix whose entries are rational functions in declared algebraically independent variables.",
        SymbolicCharacteristicPolynomialRequest,
        SymbolicCharacteristicPolynomialResult,
        _run_characteristic,
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
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
    ),
    symbolic_matrix_operation(
        "matrix.symbolic.eigenvalues.compute",
        "Compute exact symbolic eigenvalues",
        "Compute the exact eigenvalues with algebraic multiplicities of a square symbolic matrix using SymPy's eigenvals. Entries may be rational functions in declared algebraically independent variables; eigenvalues are returned as canonical SymPy expression strings.",
        SymbolicCharacteristicPolynomialRequest,
        SymbolicEigenvaluesResult,
        _run_eigenvalues,
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
                        "entries": [
                            [
                                _rational_function((), (1, ())),
                                _rational_function((), (2, ())),
                            ],
                            [
                                _rational_function((), (3, ())),
                                _rational_function((), (4, ())),
                            ],
                        ],
                    }
                },
            ),
        ),
    ),
    symbolic_matrix_operation(
        "matrix.symbolic.linear_system.solve",
        "Classify and solve a symbolic linear system over QQ(t_1, ..., t_n)",
        (
            "Classify one bounded system A x = b over the rational-function "
            "field QQ(t_1, ..., t_n) as UNIQUE, NON_UNIQUE, or INCONSISTENT. "
            "For a unique system, return the exact solution vector. For a "
            "non-unique consistent system, return a particular solution and nullspace "
            "basis. The declared parameters are algebraically independent: the "
            "result is the generic solution, not a case split over parameter "
            "specializations. Backed by SymPy symbolic linear algebra."
        ),
        SymbolicLinearSystemRequest,
        SymbolicLinearSystemResult,
        _run_linear_system,
        "matrix",
        "symbolic",
        "linear-system",
        "rational-function-field",
        "exact",
        examples=(_LINEAR_SYSTEM_EXAMPLE,),
    ),
)


__all__ = ["TOOLS"]
