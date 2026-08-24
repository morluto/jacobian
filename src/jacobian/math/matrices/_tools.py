"""Exact matrix operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices._operation_models import (
    CharacteristicPolynomialResult,
    IntegerMatrixRequest,
    MatrixAdjugateResult,
    MatrixDeterminantRequest,
    MatrixDeterminantResult,
    MatrixInverseResult,
    MatrixKroneckerProductRequest,
    MatrixKroneckerProductResult,
    MatrixPartialTraceRequest,
    MatrixPartialTraceResult,
    MatrixPermanentResult,
    MatrixProductResult,
    MatrixRankRequest,
    MatrixRankResult,
    MatrixTraceResult,
    NonsingularIntegerMatrixRequest,
    NullspaceResult,
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RrefResult,
    SquareIntegerMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.math.matrices._operations import (
    compute_adjugate,
    compute_characteristic_polynomial,
    compute_determinant,
    compute_inverse,
    compute_kronecker_product,
    compute_nullspace,
    compute_partial_trace,
    compute_permanent,
    compute_product,
    compute_rank,
    compute_rational_linear_solve,
    compute_rref,
    compute_smith_normal_form,
    compute_trace,
)
from jacobian.math.matrices.values import SmithNormalForm


def matrix_operation[
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


MATRIX_DETERMINANT_COMPUTE = matrix_operation(
    "matrix.determinant.compute",
    "Compute an exact rational matrix determinant (det)",
    "Compute the determinant of one square matrix over QQ through order 64 with SymPy's exact Bareiss algorithm.",
    MatrixDeterminantRequest,
    MatrixDeterminantResult,
    compute_determinant,
    "matrix",
    "determinant",
    "exact-rational",
    examples=(
        example(
            "determinant_minus_six",
            "Compute the determinant of [[0, 2], [3, 4]].",
            {
                "matrix": {
                    "entries": [
                        [
                            {"num": "0", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                        [
                            {"num": "3", "den": "1"},
                            {"num": "4", "den": "1"},
                        ],
                    ]
                }
            },
        ),
        example(
            "determinant_3x3_identity",
            "Compute the determinant of a 3x3 identity (1); the matrix must be square (rows == columns).",
            {
                "matrix": {
                    "entries": [
                        [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    ]
                }
            },
        ),
    ),
    version="3",
)

MATRIX_OPERATIONS = (
    MATRIX_DETERMINANT_COMPUTE,
    matrix_operation(
        "matrix.rank.compute",
        "Compute exact rational matrix rank",
        "Compute the rank and RREF pivot columns of one rectangular matrix over QQ.",
        MatrixRankRequest,
        MatrixRankResult,
        compute_rank,
        "matrix",
        "rank",
        "exact-rational",
        examples=(
            example(
                "rank_three_by_four",
                "Compute rank and pivots of a rectangular rational matrix.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "2", "den": "1"},
                                {"num": "3", "den": "1"},
                                {"num": "4", "den": "1"},
                            ],
                            [
                                {"num": "2", "den": "1"},
                                {"num": "4", "den": "1"},
                                {"num": "6", "den": "1"},
                                {"num": "8", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        ]
                    }
                },
            ),
        ),
        version="2",
    ),
    matrix_operation(
        "matrix.rational_linear_system.solve",
        "Solve an exact rational linear system",
        "Classify and solve a bounded square system Ax=b over QQ, returning a "
        "unique solution only when one exists.",
        RationalLinearSolveRequest,
        RationalLinearSolveResult,
        compute_rational_linear_solve,
        "matrix",
        "linear-system",
        "exact-rational",
        examples=(
            example(
                "solve_identity_system",
                "Solve a 2x2 identity linear system.",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ]
                    },
                    "rhs": [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                },
            ),
            example(
                "solve_3x3_diagonal",
                "Solve a 3x3 diagonal system; the matrix must be square and rhs length must match its order.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "2", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "3", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "4", "den": "1"},
                            ],
                        ]
                    },
                    "rhs": [
                        {"num": "4", "den": "1"},
                        {"num": "6", "den": "1"},
                        {"num": "8", "den": "1"},
                    ],
                },
            ),
        ),
        version="3",
    ),
    matrix_operation(
        "matrix.adjugate.compute",
        "Compute an exact matrix adjugate",
        "Compute the classical adjugate of a square integer matrix.",
        SquareIntegerMatrixRequest,
        MatrixAdjugateResult,
        compute_adjugate,
        "matrix",
        "adjugate",
        "exact-integer",
        examples=(
            example(
                "adjugate_two_by_two",
                "Compute the adjugate of a 2x2 integer matrix.",
                {"matrix": {"entries": [["1", "2"], ["3", "4"]]}},
            ),
            example(
                "adjugate_3x3_diagonal",
                "Compute the adjugate of a 3x3 diagonal matrix; the matrix must be square.",
                {
                    "matrix": {
                        "entries": [["2", "0", "0"], ["0", "3", "0"], ["0", "0", "4"]]
                    }
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.inverse.compute",
        "Compute the exact inverse of an integer matrix",
        "Compute the rational two-sided inverse of a nonsingular square matrix.",
        NonsingularIntegerMatrixRequest,
        MatrixInverseResult,
        compute_inverse,
        "matrix",
        "inverse",
        "exact-rational",
        examples=(
            example(
                "inverse_two_by_two",
                "Compute the inverse of a nonsingular 2x2 integer matrix.",
                {"matrix": {"entries": [["1", "2"], ["3", "4"]]}},
            ),
            example(
                "inverse_diagonal_3x3",
                "Compute the inverse of a 3x3 diagonal matrix; the matrix must be square and nonsingular.",
                {
                    "matrix": {
                        "entries": [["2", "0", "0"], ["0", "3", "0"], ["0", "0", "4"]]
                    }
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.trace.compute",
        "Compute the exact trace of an integer matrix",
        "Compute the sum of the diagonal entries of a square integer matrix.",
        SquareIntegerMatrixRequest,
        MatrixTraceResult,
        compute_trace,
        "matrix",
        "trace",
        "exact-integer",
        examples=(
            example(
                "trace_two_by_two",
                "Compute the trace of a 2x2 integer matrix.",
                {"matrix": {"entries": [["1", "2"], ["3", "4"]]}},
            ),
            example(
                "trace_diagonal_3x3",
                "Compute the trace (6) of a 3x3 diagonal matrix; the matrix must be square.",
                {
                    "matrix": {
                        "entries": [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "3"]]
                    }
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.multiply.compute",
        "Multiply two exact rational matrices",
        (
            "Compute the standard row-by-column product of two compatible bounded "
            "matrices over QQ, with the operand shapes bound in the result. Equal "
            "operands give the exact self-product or matrix square."
        ),
        RationalMatrixProductRequest,
        MatrixProductResult,
        compute_product,
        "matrix",
        "matrix-multiplication",
        "product",
        "self-product",
        "matrix-square",
        "zero-matrix",
        "matrix-identity",
        "exact-rational",
        examples=(
            example(
                "multiply_rectangular_matrices",
                "Multiply a 2x3 matrix by a 3x2 matrix over QQ.",
                {
                    "left": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "2", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ]
                    },
                    "right": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ]
                    },
                },
            ),
            example(
                "multiply_square_matrices",
                "Multiply two 2x2 matrices; the left column count must equal the right row count.",
                {
                    "left": {
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ]
                    },
                    "right": {
                        "entries": [
                            [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
                        ]
                    },
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.normal_form.rref.compute",
        "Compute exact reduced row echelon form",
        "Compute the unique reduced row echelon form over QQ.",
        RationalMatrixRequest,
        RrefResult,
        compute_rref,
        "matrix",
        "rref",
        "exact-rational",
        examples=(
            example(
                "rref_two_by_two",
                "Compute RREF of a rational matrix.",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                            [{"num": "2", "den": "1"}, {"num": "4", "den": "1"}],
                        ]
                    }
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.nullspace.compute",
        "Compute a canonical exact nullspace or relation basis",
        (
            "Compute the RREF fundamental basis of the right nullspace over QQ. "
            "When columns are ordered vectors, the result gives their rank and "
            "every exact rational linear dependency coefficient."
        ),
        RationalMatrixRequest,
        NullspaceResult,
        compute_nullspace,
        "matrix",
        "nullspace",
        "kernel",
        "linear-dependence",
        "rational-relations",
        "exact-rational",
        examples=(
            example(
                "rational_relation_among_columns",
                ("Compute every rational relation among three ordered column vectors."),
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ]
                    }
                },
            ),
        ),
        version="2",
    ),
    matrix_operation(
        "matrix.characteristic_polynomial.compute",
        "Compute an exact characteristic polynomial",
        "Compute dense coefficients of det(lambda I - A) over QQ.",
        SquareRationalMatrixRequest,
        CharacteristicPolynomialResult,
        compute_characteristic_polynomial,
        "matrix",
        "characteristic-polynomial",
        "exact-rational",
        examples=(
            example(
                "characteristic_two_by_two",
                "Compute the characteristic polynomial of a 2x2 matrix.",
                {
                    "matrix": {
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                            [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                        ]
                    }
                },
            ),
            example(
                "characteristic_diagonal_3x3",
                "Compute the characteristic polynomial of a diagonal 3x3; the matrix must be square.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "2", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "3", "den": "1"},
                            ],
                        ]
                    }
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.normal_form.smith.compute",
        "Compute an exact Smith normal form",
        (
            "Compute the canonical diagonal Smith form over ZZ without claiming "
            "unavailable left or right transformations."
        ),
        IntegerMatrixRequest,
        SmithNormalForm,
        compute_smith_normal_form,
        "matrix",
        "smith-normal-form",
        "exact-integer",
        examples=(
            example(
                "smith_two_by_two",
                "Compute the Smith normal form of a 2x2 integer matrix.",
                {"matrix": {"entries": [["2", "4"], ["6", "8"]]}},
            ),
        ),
        version="2",
    ),
    matrix_operation(
        "matrix.permanent.compute",
        "Compute an exact matrix permanent",
        "Compute the permanent (sign-free determinant analogue) of a square rational matrix over QQ through order 64 with SymPy's exact Permanent backend.",
        SquareRationalMatrixRequest,
        MatrixPermanentResult,
        compute_permanent,
        "matrix",
        "permanent",
        "exact-rational",
        examples=(
            example(
                "permanent_two_by_two",
                "Compute the permanent of [[1, 2], [3, 4]].",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "2", "den": "1"},
                            ],
                            [
                                {"num": "3", "den": "1"},
                                {"num": "4", "den": "1"},
                            ],
                        ]
                    }
                },
            ),
            example(
                "permanent_identity_3x3",
                "Compute the permanent (1) of a 3x3 identity; the matrix must be square.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ]
                    }
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.kronecker_product.compute",
        "Compute an exact Kronecker product",
        "Compute the Kronecker (tensor) product of two bounded rational matrices over QQ.",
        MatrixKroneckerProductRequest,
        MatrixKroneckerProductResult,
        compute_kronecker_product,
        "matrix",
        "kronecker-product",
        "tensor-product",
        "exact-rational",
        examples=(
            example(
                "kronecker_two_by_two",
                "Compute the Kronecker product of two 2x2 matrices.",
                {
                    "left": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        ]
                    },
                    "right": {
                        "entries": [
                            [
                                {"num": "2", "den": "1"},
                                {"num": "3", "den": "1"},
                            ],
                            [
                                {"num": "4", "den": "1"},
                                {"num": "5", "den": "1"},
                            ],
                        ]
                    },
                },
            ),
        ),
    ),
    matrix_operation(
        "matrix.partial_trace.compute",
        "Compute an exact partial trace over a Kronecker factor",
        "Compute the partial trace over the first (traced) subsystem of a composite matrix A (x) B stored in row-major block order over QQ.",
        MatrixPartialTraceRequest,
        MatrixPartialTraceResult,
        compute_partial_trace,
        "matrix",
        "partial-trace",
        "tensor",
        "exact-rational",
        examples=(
            example(
                "partial_trace_diagonal",
                "Trace out a 2x2 diagonal factor from a 4x4 Kronecker product.",
                {
                    "matrix": {
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "2", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                                {"num": "2", "den": "1"},
                            ],
                        ]
                    },
                    "traced_dimension": 2,
                    "kept_dimension": 2,
                },
            ),
        ),
    ),
)

TOOLS = MATRIX_OPERATIONS

__all__ = ["TOOLS"]
