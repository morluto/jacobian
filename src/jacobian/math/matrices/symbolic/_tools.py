"""Exact symbolic matrix operation declarations."""

from collections.abc import Callable
from typing import Any

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
from jacobian.math.matrices.symbolic._operations import (
    compute_symbolic_characteristic_polynomial,
    compute_symbolic_determinant,
    compute_symbolic_eigenvalues,
    compute_symbolic_linear_system,
    compute_symbolic_matrix_product,
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
    version: str = "2",
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
        "rational_function_schema_version": "1",
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


SYMBOLIC_MATRIX_OPERATIONS = (
    symbolic_matrix_operation(
        "matrix.symbolic.determinant.compute",
        "Compute an exact symbolic matrix determinant (det) over QQ(t_1, ..., t_n)",
        "Compute the determinant of a square matrix whose entries are rational functions in declared algebraically independent variables, using SymPy's exact fraction-free Bareiss algorithm.",
        SymbolicDeterminantRequest,
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
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
        version="2",
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
        compute_symbolic_matrix_product,
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
                    "matrix": _generic_two_by_two(),
                },
            ),
        ),
        version="2",
    ),
    symbolic_matrix_operation(
        "matrix.symbolic.eigenvalues.compute",
        "Compute exact symbolic eigenvalues",
        "Compute the exact eigenvalues with algebraic multiplicities of a square symbolic matrix using SymPy's eigenvals. Entries may be rational functions in declared algebraically independent variables; eigenvalues are returned as canonical SymPy expression strings.",
        SymbolicCharacteristicPolynomialRequest,
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
        version="2",
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
        compute_symbolic_linear_system,
        "matrix",
        "symbolic",
        "linear-system",
        "rational-function-field",
        "exact",
        examples=(_LINEAR_SYSTEM_EXAMPLE,),
    ),
)

TOOLS = SYMBOLIC_MATRIX_OPERATIONS

__all__ = ["TOOLS"]
