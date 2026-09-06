"""Exact symbolic matrix operation declarations."""

from typing import Any

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
    symbolic_characteristic_polynomial,
    symbolic_determinant,
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
    degree, coefficients = symbolic_characteristic_polynomial(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicEigenvaluesResult(
        matrix=request.matrix,
        characteristic_polynomial=coefficients,
        degree=degree,
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


_LINEAR_SYSTEM_EXAMPLE = OperationExample(
    name="symbolic_linear_system_unique",
    description="Solve [[1, t], [0, 1]] x = [t, 1] over QQ(t); solution is x = [0, 1].",
    input={
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


_SYMBOLIC_PRODUCT_EXAMPLE = OperationExample(
    name="symbolic_matrix_product",
    description="Multiply [[a, b]] by the column [[1], [1]] over QQ(a, b); both matrices must use the same ordered field and compatible inner dimension.",
    input={
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
    MathTool(
        operation_id="matrix.symbolic.determinant.compute",
        title="Compute an exact symbolic matrix determinant (det) over QQ(t_1, ..., t_n)",
        description="Compute the determinant of a square matrix whose entries are rational functions in declared algebraically independent variables, using SymPy's exact fraction-free Bareiss algorithm.",
        request_type=SymbolicDeterminantRequest,
        result_type=SymbolicDeterminantResult,
        run=_run_determinant,
        tags=("matrix", "symbolic", "determinant", "rational-function-field", "exact"),
        examples=(
            OperationExample(
                name="symbolic_determinant_two_by_two",
                description="Compute the determinant of [[a, c], [b, d]]; the matrix must be square and rectangular over declared variables.",
                input={
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.symbolic.rank.compute",
        title="Compute exact symbolic matrix rank over QQ(t_1, ..., t_n)",
        description="Compute the rank and RREF pivot columns of a rectangular matrix whose entries are rational functions in declared algebraically independent variables, using SymPy's exact row reduction.",
        request_type=SymbolicMatrixRequest,
        result_type=SymbolicRankResult,
        run=_run_rank,
        tags=("matrix", "symbolic", "rank", "rational-function-field", "exact"),
        examples=(
            OperationExample(
                name="symbolic_rank_full",
                description="Compute the rank of a 2x2 symbolic matrix; rows must be nonempty and equal length over declared variables.",
                input={
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.symbolic.multiply.compute",
        title="Multiply exact symbolic matrices over QQ(t_1, ..., t_n)",
        description=(
            "Compute the row-by-column product of two compatible symbolic matrices "
            "over one explicitly ordered rational-function field. Every product "
            "entry is returned as a canonical reduced rational function; admission "
            "bounds unreduced expansion plus cancellation-safe canonical and "
            "aggregate support, exponents, coefficients, and result before SymPy "
            "multiplication."
        ),
        request_type=SymbolicMatrixProductRequest,
        result_type=SymbolicMatrix,
        run=_run_product,
        tags=(
            "matrix",
            "symbolic",
            "matrix-multiplication",
            "product",
            "rational-function-field",
            "exact",
        ),
        examples=(_SYMBOLIC_PRODUCT_EXAMPLE,),
    ),
    MathTool(
        operation_id="matrix.symbolic.characteristic_polynomial.compute",
        title="Compute an exact symbolic characteristic polynomial",
        description="Compute the dense monic coefficients of det(lambda I - A) for a square symbolic matrix whose entries are rational functions in declared algebraically independent variables.",
        request_type=SymbolicCharacteristicPolynomialRequest,
        result_type=SymbolicCharacteristicPolynomialResult,
        run=_run_characteristic,
        tags=(
            "matrix",
            "symbolic",
            "characteristic-polynomial",
            "rational-function-field",
            "exact",
        ),
        examples=(
            OperationExample(
                name="symbolic_charpoly_two_by_two",
                description="Compute the characteristic polynomial of [[a, c], [b, d]]; the matrix must be square and rectangular.",
                input={
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.symbolic.eigenvalues.compute",
        title="Compute exact symbolic eigenvalues",
        description=(
            "Compute the exact eigenvalue claim as the characteristic polynomial "
            "of a square symbolic matrix over its declared rational-function "
            "field. Individual algebraic roots remain represented by this typed "
            "polynomial rather than backend expression strings."
        ),
        request_type=SymbolicCharacteristicPolynomialRequest,
        result_type=SymbolicEigenvaluesResult,
        run=_run_eigenvalues,
        tags=("matrix", "symbolic", "eigenvalues", "rational-function-field", "exact"),
        examples=(
            OperationExample(
                name="symbolic_eigenvalues_two_by_two",
                description="Compute the exact eigenvalues of [[1, 2], [3, 4]]; the matrix must be square and rectangular.",
                input={
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
    MathTool(
        operation_id="matrix.symbolic.linear_system.solve",
        title="Classify and solve a symbolic linear system over QQ(t_1, ..., t_n)",
        description=(
            "Classify one bounded system A x = b over the rational-function "
            "field QQ(t_1, ..., t_n) as UNIQUE, NON_UNIQUE, or INCONSISTENT. "
            "For a unique system, return the exact solution vector. For a "
            "non-unique consistent system, return a particular solution and nullspace "
            "basis. The declared parameters are algebraically independent: the "
            "result is the generic solution, not a case split over parameter "
            "specializations. Backed by SymPy symbolic linear algebra."
        ),
        request_type=SymbolicLinearSystemRequest,
        result_type=SymbolicLinearSystemResult,
        run=_run_linear_system,
        tags=(
            "matrix",
            "symbolic",
            "linear-system",
            "rational-function-field",
            "exact",
        ),
        examples=(_LINEAR_SYSTEM_EXAMPLE,),
    ),
)


__all__ = ["TOOLS"]
