"""Exact matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from jacobian.contracts.matrix_operations import (
    CharacteristicPolynomialResult,
    IntegerMatrixRequest,
    MatrixAdjugateResult,
    MatrixDeterminantRequest,
    MatrixDeterminantResult,
    MatrixInverseResult,
    MatrixProductResult,
    MatrixRankRequest,
    MatrixRankResult,
    MatrixTraceResult,
    NullspaceResult,
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RrefResult,
    SquareIntegerMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.contracts.operations import (
    OperationDiagnostic,
    OperationExample,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.domains.matrix_lattice.operations import (
    compute_adjugate,
    compute_characteristic_polynomial,
    compute_determinant,
    compute_inverse,
    compute_nullspace,
    compute_product,
    compute_rank,
    compute_rational_linear_solve,
    compute_rref,
    compute_smith_normal_form,
    compute_trace,
)
from jacobian.math.matrices.values import SmithNormalForm
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operation_ports import OutputPort
from jacobian.operations import (
    OperationAbortError,
    OperationRefusalError,
)


def matrix_operation[
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
    output_ports: tuple[OutputPort[Any], ...] = (),
    version: str = "1",
) -> OperationDeclaration[RequestT, ResultT]:
    def implementation(request: RequestT) -> ResultT:
        try:
            return operation(request)
        except ValidationError as exc:
            raise OperationAbortError(
                ExecutionStatus.ERROR,
                OperationDiagnostic(
                    code="MATRIX_OUTPUT_LIMIT_EXCEEDED",
                    stage="matrix_result_validation",
                    message=(
                        "The exact matrix result exceeded its bounded output "
                        f"contract: {exc}"
                    ),
                    hint=(
                        "Reduce the matrix dimension or scalar size; no result "
                        "artifact was retained."
                    ),
                ),
            ) from exc
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise OperationRefusalError(
                OperationDiagnostic(
                    code="MATRIX_OPERATION_NOT_APPLICABLE",
                    stage="matrix_computation",
                    message=str(exc),
                    hint="Check the operation's matrix-domain and shape preconditions.",
                )
            ) from exc

    return inline_operation(
        OperationDeclaration(
            operation_id=operation_id,
            version=version,
            title=title,
            description=description,
            request_type=request_model,
            result_type=result_model,
            execute=implementation,
            tags=tags,
            examples=examples,
        ),
        output_ports=output_ports,
    )


MATRIX_OPERATIONS = (
    matrix_operation(
        "matrix.determinant.compute",
        "Compute an exact rational matrix determinant",
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
        ),
        version="3",
    ),
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
        "Compute the unique solution to a bounded square system Ax=b over QQ.",
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
        ),
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
        ),
    ),
    matrix_operation(
        "matrix.inverse.compute",
        "Compute the exact inverse of an integer matrix",
        "Compute the rational two-sided inverse of a nonsingular square matrix.",
        SquareIntegerMatrixRequest,
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
        output_ports=(OutputPort(name="smith_form", value_type=SmithNormalForm),),
        version="2",
    ),
)
