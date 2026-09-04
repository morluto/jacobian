"""Exact matrix operation declarations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices._operation_models import (
    CharacteristicPolynomialRequest,
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
    MatrixPermanentRequest,
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
)
from jacobian.math.matrices.operations import (
    adjugate_result,
    characteristic_polynomial_result,
    determinant_result,
    inverse_result,
    kronecker_product_result,
    nullspace_result,
    partial_trace_result,
    permanent_result,
    product_result,
    rank_result,
    rational_linear_solve_result,
    rref_result,
    smith_normal_form_result,
    trace_result,
)
from jacobian.math.matrices.values import SmithNormalForm


def compute_determinant(request: MatrixDeterminantRequest) -> MatrixDeterminantResult:
    return determinant_result(request.matrix)


def compute_adjugate(request: SquareIntegerMatrixRequest) -> MatrixAdjugateResult:
    return adjugate_result(request.matrix)


def compute_trace(request: SquareIntegerMatrixRequest) -> MatrixTraceResult:
    return trace_result(request.matrix)


def compute_product(request: RationalMatrixProductRequest) -> MatrixProductResult:
    return product_result(request.left, request.right)


def compute_kronecker_product(
    request: MatrixKroneckerProductRequest,
) -> MatrixKroneckerProductResult:
    return kronecker_product_result(request.left, request.right)


def compute_rank(request: MatrixRankRequest) -> MatrixRankResult:
    return rank_result(request.matrix)


def compute_rational_linear_solve(
    request: RationalLinearSolveRequest,
) -> RationalLinearSolveResult:
    return rational_linear_solve_result(request.matrix, request.rhs)


def compute_inverse(request: NonsingularIntegerMatrixRequest) -> MatrixInverseResult:
    return inverse_result(request.matrix)


def compute_rref(request: RationalMatrixRequest) -> RrefResult:
    return rref_result(request.matrix)


def compute_nullspace(request: RationalMatrixRequest) -> NullspaceResult:
    return nullspace_result(request.matrix)


def compute_characteristic_polynomial(
    request: CharacteristicPolynomialRequest,
) -> CharacteristicPolynomialResult:
    return characteristic_polynomial_result(request.matrix)


def compute_smith_normal_form(request: IntegerMatrixRequest) -> SmithNormalForm:
    return smith_normal_form_result(request.matrix)


def compute_permanent(request: MatrixPermanentRequest) -> MatrixPermanentResult:
    return permanent_result(request.matrix)


def compute_partial_trace(
    request: MatrixPartialTraceRequest,
) -> MatrixPartialTraceResult:
    return partial_trace_result(
        request.matrix,
        request.traced_dimension,
        request.kept_dimension,
    )


MATRIX_DETERMINANT_COMPUTE = MathTool(
    operation_id="matrix.determinant.compute",
    title="Compute an exact rational matrix determinant (det)",
    description="Compute the determinant of one square matrix over QQ through order 128 with FLINT's exact dense rational kernel, subject to scalar-work and result-height bounds.",
    request_type=MatrixDeterminantRequest,
    result_type=MatrixDeterminantResult,
    run=compute_determinant,
    tags=("matrix", "determinant", "exact-rational"),
    examples=(
        OperationExample(
            name="determinant_minus_six",
            description="Compute the determinant of [[0, 2], [3, 4]].",
            input={
                "matrix": {
                    "domain": "QQ",
                    "entries": [
                        [
                            {"num": "0", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                        [
                            {"num": "3", "den": "1"},
                            {"num": "4", "den": "1"},
                        ],
                    ],
                }
            },
        ),
        OperationExample(
            name="determinant_3x3_identity",
            description="Compute the determinant of a 3x3 identity (1); the matrix must be square (rows == columns).",
            input={
                "matrix": {
                    "domain": "QQ",
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
                    ],
                }
            },
        ),
    ),
)

TOOLS = (
    MATRIX_DETERMINANT_COMPUTE,
    MathTool(
        operation_id="matrix.adjugate.compute",
        title="Compute an exact matrix adjugate",
        description="Compute the classical adjugate of a square integer matrix.",
        request_type=SquareIntegerMatrixRequest,
        result_type=MatrixAdjugateResult,
        run=compute_adjugate,
        tags=("matrix", "adjugate", "exact-integer"),
        examples=(
            OperationExample(
                name="adjugate_two_by_two",
                description="Compute the adjugate of [[1, 2], [3, 4]].",
                input={"matrix": {"entries": [["1", "2"], ["3", "4"]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.trace.compute",
        title="Compute the exact trace of an integer matrix",
        description="Compute the sum of the diagonal entries of a square integer matrix.",
        request_type=SquareIntegerMatrixRequest,
        result_type=MatrixTraceResult,
        run=compute_trace,
        tags=("matrix", "trace", "exact-integer"),
        examples=(
            OperationExample(
                name="trace_two_by_two",
                description="Compute the trace of [[1, 2], [3, 4]].",
                input={"matrix": {"entries": [["1", "2"], ["3", "4"]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.multiply.compute",
        title="Multiply two exact rational matrices",
        description="Compute the standard row-by-column product of two compatible bounded matrices over QQ.",
        request_type=RationalMatrixProductRequest,
        result_type=MatrixProductResult,
        run=compute_product,
        tags=("matrix", "matrix-multiplication", "product", "exact-rational"),
        examples=(
            OperationExample(
                name="multiply_square_matrices",
                description="Multiply two 2x2 matrices over QQ.",
                input={
                    "left": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                    "right": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.kronecker_product.compute",
        title="Compute an exact Kronecker product",
        description="Compute the Kronecker product of two bounded rational matrices over QQ.",
        request_type=MatrixKroneckerProductRequest,
        result_type=MatrixKroneckerProductResult,
        run=compute_kronecker_product,
        tags=("matrix", "kronecker-product", "tensor", "exact-rational"),
        examples=(
            OperationExample(
                name="kronecker_identity",
                description="Compute the Kronecker product of two 2x2 identity matrices.",
                input={
                    "left": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                    "right": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.rank.compute",
        title="Compute exact rational matrix rank",
        description="Compute the rank and RREF pivot columns of one dense or coordinate-sparse rectangular matrix over QQ. Dense matrices are admitted through 64 axes; sparse matrices retain declared axes through 8192 and are admitted by connected support components, scalar work, intermediate height, and exact output size.",
        request_type=MatrixRankRequest,
        result_type=MatrixRankResult,
        run=compute_rank,
        tags=("matrix", "rank", "exact-rational"),
        examples=(
            OperationExample(
                name="rank_three_by_four",
                description="Compute rank and pivots of a rectangular rational matrix.",
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
                    }
                },
            ),
            OperationExample(
                name="rank_sparse_last_column",
                description="Compute rank while retaining a sparse matrix's declared column axis.",
                input={
                    "matrix": {
                        "row_count": 1,
                        "column_count": 128,
                        "entries": [
                            {
                                "row": 0,
                                "column": 127,
                                "value": {"num": "1", "den": "1"},
                            }
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.rational_linear_system.solve",
        title="Classify a square rational linear system",
        description="Classify and solve a bounded square system Ax=b over QQ, returning a "
        "unique solution only when one exists.",
        request_type=RationalLinearSolveRequest,
        result_type=RationalLinearSolveResult,
        run=compute_rational_linear_solve,
        tags=("matrix", "linear-system", "exact-rational"),
        examples=(
            OperationExample(
                name="solve_identity_system",
                description="Solve a 2x2 identity linear system.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                    "rhs": [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                },
            ),
            OperationExample(
                name="solve_3x3_diagonal",
                description="Solve a 3x3 diagonal system; the matrix must be square and rhs length must match its order.",
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
                    },
                    "rhs": [
                        {"num": "4", "den": "1"},
                        {"num": "6", "den": "1"},
                        {"num": "8", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.inverse.compute",
        title="Compute the exact inverse of an integer matrix",
        description="Compute the rational two-sided inverse of a nonsingular square matrix.",
        request_type=NonsingularIntegerMatrixRequest,
        result_type=MatrixInverseResult,
        run=compute_inverse,
        tags=("matrix", "inverse", "exact-rational"),
        examples=(
            OperationExample(
                name="inverse_two_by_two",
                description="Compute the inverse of a nonsingular 2x2 integer matrix.",
                input={"matrix": {"entries": [["1", "2"], ["3", "4"]]}},
            ),
            OperationExample(
                name="inverse_diagonal_3x3",
                description="Compute the inverse of a 3x3 diagonal matrix; the matrix must be square and nonsingular.",
                input={
                    "matrix": {
                        "entries": [["2", "0", "0"], ["0", "3", "0"], ["0", "0", "4"]]
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.normal_form.rref.compute",
        title="Compute exact reduced row echelon form",
        description="Compute the unique reduced row echelon form over QQ through 64 rows and columns, subject to scalar-work and result-height bounds.",
        request_type=RationalMatrixRequest,
        result_type=RrefResult,
        run=compute_rref,
        tags=("matrix", "rref", "exact-rational"),
        examples=(
            OperationExample(
                name="rref_two_by_two",
                description="Compute RREF of a rational matrix.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                            [{"num": "2", "den": "1"}, {"num": "4", "den": "1"}],
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.nullspace.compute",
        title="Compute a canonical exact nullspace or relation basis",
        description=(
            "Compute the RREF fundamental basis of the right nullspace over QQ through 64 rows and columns, subject to scalar-work and result-height bounds. "
            "When columns are ordered vectors, the result gives their rank and "
            "every exact rational linear dependency coefficient."
        ),
        request_type=RationalMatrixRequest,
        result_type=NullspaceResult,
        run=compute_nullspace,
        tags=(
            "matrix",
            "nullspace",
            "kernel",
            "linear-dependence",
            "rational-relations",
            "exact-rational",
        ),
        examples=(
            OperationExample(
                name="rational_relation_among_columns",
                description=(
                    "Compute every rational relation among three ordered column vectors."
                ),
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.characteristic_polynomial.compute",
        title="Compute an exact characteristic polynomial",
        description="Compute dense coefficients of det(lambda I - A) over QQ.",
        request_type=CharacteristicPolynomialRequest,
        result_type=CharacteristicPolynomialResult,
        run=compute_characteristic_polynomial,
        tags=("matrix", "characteristic-polynomial", "exact-rational"),
        examples=(
            OperationExample(
                name="characteristic_two_by_two",
                description="Compute the characteristic polynomial of a 2x2 matrix.",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                            [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                        ],
                    }
                },
            ),
            OperationExample(
                name="characteristic_diagonal_3x3",
                description="Compute the characteristic polynomial of a diagonal 3x3; the matrix must be square.",
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.normal_form.smith.compute",
        title="Compute an exact Smith normal form",
        description=(
            "Compute the canonical diagonal Smith form over ZZ through 64 rows and columns, subject to scalar-work and result-height bounds, without claiming unavailable left or right transformations."
        ),
        request_type=IntegerMatrixRequest,
        result_type=SmithNormalForm,
        run=compute_smith_normal_form,
        tags=("matrix", "smith-normal-form", "exact-integer"),
        examples=(
            OperationExample(
                name="smith_two_by_two",
                description="Compute the Smith normal form of a 2x2 integer matrix.",
                input={"matrix": {"entries": [["2", "4"], ["6", "8"]]}},
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.permanent.compute",
        title="Compute an exact matrix permanent",
        description="Compute the permanent (sign-free determinant analogue) of a square rational matrix over QQ through order 12. The owner charges SymPy's Ryser algorithm against its 4,096-subset budget.",
        request_type=MatrixPermanentRequest,
        result_type=MatrixPermanentResult,
        run=compute_permanent,
        tags=("matrix", "permanent", "exact-rational"),
        examples=(
            OperationExample(
                name="permanent_two_by_two",
                description="Compute the permanent of [[1, 2], [3, 4]].",
                input={
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [
                                {"num": "1", "den": "1"},
                                {"num": "2", "den": "1"},
                            ],
                            [
                                {"num": "3", "den": "1"},
                                {"num": "4", "den": "1"},
                            ],
                        ],
                    }
                },
            ),
            OperationExample(
                name="permanent_identity_3x3",
                description="Compute the permanent (1) of a 3x3 identity; the matrix must be square.",
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.partial_trace.compute",
        title="Compute an exact partial trace over a Kronecker factor",
        description="Compute the partial trace over the first (traced) subsystem of a composite matrix A (x) B stored in row-major block order over QQ.",
        request_type=MatrixPartialTraceRequest,
        result_type=MatrixPartialTraceResult,
        run=compute_partial_trace,
        tags=("matrix", "partial-trace", "tensor", "exact-rational"),
        examples=(
            OperationExample(
                name="partial_trace_diagonal",
                description="Trace out a 2x2 diagonal factor from a 4x4 Kronecker product.",
                input={
                    "matrix": {
                        "domain": "QQ",
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
                        ],
                    },
                    "traced_dimension": 2,
                    "kept_dimension": 2,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
