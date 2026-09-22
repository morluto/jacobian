"""Public declarations for exact tropical operations."""
# ruff: noqa: F405

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.polynomials.tropical._models import *  # noqa: F403
from jacobian.math.polynomials.tropical.operations import *  # noqa: F403
from jacobian.math.polynomials.tropical.values import *  # noqa: F403


def _s() -> dict[str, str]:
    return {"convention": "MIN_PLUS", "base": "ZZ"}


def _finite(n: int) -> dict[str, object]:
    return {"semiring": _s(), "kind": "FINITE", "value": {"num": str(n), "den": "1"}}


def _vector(values: tuple[int, ...], axis: tuple[str, ...]) -> dict[str, object]:
    return {
        "semiring": _s(),
        "axis": list(axis),
        "entries": [_finite(v) for v in values],
    }


def _poly(values: tuple[tuple[tuple[int, ...], int], ...]) -> dict[str, object]:
    return {
        "semiring": _s(),
        "variables": ["x"],
        "terms": [{"exponents": list(e), "coefficient": _finite(c)} for e, c in values],
    }


def _matrix(values: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    return {
        "semiring": _s(),
        "row_axis": ["a", "b"],
        "column_axis": ["a", "b"],
        "entries": [[_finite(v) for v in row] for row in values],
    }


def compute_scalar_add(request: ScalarAddRequest) -> ScalarAddResult:
    result, branch, case = tropical_scalar_add(
        request.semiring, request.left, request.right
    )
    return ScalarAddResult._from_kernel(
        request, result=result, branch=branch, infinity_case=case
    )


def compute_scalar_multiply(request: ScalarBinaryRequest) -> ScalarResult:
    return ScalarResult._from_kernel(
        tropical_scalar_multiply(request.left, request.right)
    )


def compute_scalar_power(request: ScalarPowerRequest) -> ScalarResult:
    return ScalarResult._from_kernel(
        tropical_scalar_power(request.scalar, request.exponent)
    )


def compute_vector_add(request: VectorBinaryRequest) -> VectorResult:
    return VectorResult._from_kernel(tropical_vector_add(request.left, request.right))


def compute_vector_scale(request: VectorScaleRequest) -> VectorResult:
    return VectorResult._from_kernel(
        tropical_vector_scale(request.scalar, request.vector)
    )


def compute_polynomial_add(request: PolynomialBinaryRequest) -> PolynomialResult:
    return PolynomialResult._from_kernel(
        tropical_polynomial_add(request.left, request.right)
    )


def compute_polynomial_multiply(request: PolynomialBinaryRequest) -> PolynomialResult:
    return PolynomialResult._from_kernel(
        tropical_polynomial_multiply(request.left, request.right)
    )


def compute_polynomial_evaluate(
    request: PolynomialEvaluateRequest,
) -> PolynomialEvaluateResult:
    value, active = tropical_polynomial_evaluate(request.polynomial, request.point)
    return PolynomialEvaluateResult._from_kernel(
        polynomial=request.polynomial,
        point=request.point,
        value=value,
        active_exponents=active,
    )


def compute_matrix_multiply(request: MatrixMultiplyRequest) -> MatrixResult:
    return MatrixResult._from_kernel(
        tropical_matrix_multiply(request.left, request.right)
    )


def compute_matrix_power(request: MatrixPowerRequest) -> MatrixResult:
    return MatrixResult._from_kernel(
        tropical_matrix_power(request.matrix, request.exponent)
    )


def compute_finite_power_sum(
    request: MatrixFinitePowerSumRequest,
) -> FinitePowerSumResult:
    result, winners = tropical_matrix_finite_power_sum(
        request.matrix, request.max_power
    )
    return FinitePowerSumResult._from_kernel(
        source_matrix=request.matrix,
        max_power=request.max_power,
        matrix=result,
        winning_lengths=winners,
    )


def compute_assignment(request: MatrixAssignmentRequest) -> AssignmentResult:
    value, perms = tropical_assignment_profile(request.matrix)
    return AssignmentResult._from_kernel(
        matrix=request.matrix, value=value, permutations=perms
    )


_EX = OperationExample(
    name="min_plus_finite",
    description="Compute min(3,5)=3; both finite scalars must carry the same MIN_PLUS semiring.",
    input={"semiring": _s(), "left": _finite(3), "right": _finite(5)},
)
TOOLS: MathTools = (
    MathTool(
        operation_id="tropical.scalar.add.compute",
        title="Add tropical scalars",
        description="Compute exact min-plus or max-plus addition.",
        request_type=ScalarAddRequest,
        result_type=ScalarAddResult,
        run=compute_scalar_add,
        tags=("tropical", "semiring", "exact"),
        examples=(_EX,),
    ),
    MathTool(
        operation_id="tropical.scalar.multiply.compute",
        title="Multiply tropical scalars",
        description="Compute exact tropical multiplication, ordinary addition with licensed infinity absorption.",
        request_type=ScalarBinaryRequest,
        result_type=ScalarResult,
        run=compute_scalar_multiply,
        tags=("tropical", "semiring", "exact"),
        examples=(
            OperationExample(
                name="multiply",
                description="Compute 3 times 5 in MIN_PLUS as 8; scalars must share a semiring.",
                input={"semiring": _s(), "left": _finite(3), "right": _finite(5)},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.scalar.power.compute",
        title="Power a tropical scalar",
        description="Compute a bounded exact tropical power.",
        request_type=ScalarPowerRequest,
        result_type=ScalarResult,
        run=compute_scalar_power,
        tags=("tropical", "semiring", "exact"),
        examples=(
            OperationExample(
                name="power",
                description="Compute 3 to tropical power 2 as 6; exponent must be nonnegative.",
                input={"scalar": _finite(3), "exponent": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.vector.add.compute",
        title="Add tropical vectors",
        description="Compute coordinatewise tropical addition on one labelled axis.",
        request_type=VectorBinaryRequest,
        result_type=VectorResult,
        run=compute_vector_add,
        tags=("tropical", "vector", "exact"),
        examples=(
            OperationExample(
                name="vector_add",
                description="Compute coordinatewise min of two vectors; vectors must share their labelled axis.",
                input={
                    "left": _vector((1, 4), ("x", "y")),
                    "right": _vector((2, 3), ("x", "y")),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.vector.scalar_multiply.compute",
        title="Scale a tropical vector",
        description="Add one finite tropical scalar to every vector coordinate.",
        request_type=VectorScaleRequest,
        result_type=VectorResult,
        run=compute_vector_scale,
        tags=("tropical", "vector", "exact"),
        examples=(
            OperationExample(
                name="vector_scale",
                description="Add scalar 2 to every vector coordinate; scalar and vector must share their semiring.",
                input={"scalar": _finite(2), "vector": _vector((1, 4), ("x", "y"))},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.add.compute",
        title="Add formal tropical polynomials",
        description="Combine sparse formal tropical polynomials by exponentwise tropical addition.",
        request_type=PolynomialBinaryRequest,
        result_type=PolynomialResult,
        run=compute_polynomial_add,
        tags=("tropical", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="poly_add",
                description="Add two sparse tropical polynomials; terms must use one shared variable axis.",
                input={
                    "left": _poly((((0,), 1), ((1,), 3))),
                    "right": _poly((((0,), 2), ((1,), 4))),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.multiply.compute",
        title="Multiply formal tropical polynomials",
        description="Compute sparse tropical convolution exactly.",
        request_type=PolynomialBinaryRequest,
        result_type=PolynomialResult,
        run=compute_polynomial_multiply,
        tags=("tropical", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="poly_multiply",
                description="Multiply two sparse tropical polynomials; exponents are nonnegative and share an axis.",
                input={
                    "left": _poly((((0,), 0), ((1,), 1))),
                    "right": _poly((((0,), 0), ((1,), 2))),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.polynomial.evaluate.compute",
        title="Evaluate a tropical polynomial",
        description="Evaluate every monomial exactly and return all active exponents.",
        request_type=PolynomialEvaluateRequest,
        result_type=PolynomialEvaluateResult,
        run=compute_polynomial_evaluate,
        tags=("tropical", "polynomial", "exact"),
        examples=(
            OperationExample(
                name="poly_eval",
                description="Evaluate a sparse tropical polynomial at a labelled point; point axis must match variables.",
                input={
                    "polynomial": _poly((((0,), 0), ((1,), 1))),
                    "point": _vector((2,), ("x",)),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.multiply.compute",
        title="Multiply tropical matrices",
        description="Compute exact labelled-axis tropical matrix multiplication.",
        request_type=MatrixMultiplyRequest,
        result_type=MatrixResult,
        run=compute_matrix_multiply,
        tags=("tropical", "matrix", "exact"),
        examples=(
            OperationExample(
                name="matrix_product",
                description="Multiply two 2 by 2 tropical matrices; inner labelled axes must agree.",
                input={
                    "left": _matrix(
                        ((0, 1), (2, 3)),
                    ),
                    "right": _matrix(
                        ((1, 2), (3, 4)),
                    ),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.power.compute",
        title="Power a tropical matrix",
        description="Compute a finite exact tropical matrix power.",
        request_type=MatrixPowerRequest,
        result_type=MatrixResult,
        run=compute_matrix_power,
        tags=("tropical", "matrix", "exact"),
        examples=(
            OperationExample(
                name="matrix_power",
                description="Compute a finite power of a square tropical matrix; row and column axes must agree.",
                input={
                    "matrix": _matrix(
                        ((0, 1), (2, 3)),
                    ),
                    "exponent": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.finite_power_sum.compute",
        title="Compute a finite tropical matrix power sum",
        description="Return the exact finite sum I plus A through A^N and winning path lengths; no infinite closure is claimed.",
        request_type=MatrixFinitePowerSumRequest,
        result_type=FinitePowerSumResult,
        run=compute_finite_power_sum,
        tags=("tropical", "matrix", "finite", "exact"),
        examples=(
            OperationExample(
                name="finite_power_sum",
                description="Compute a finite tropical power sum; the matrix must be square and the bound is finite.",
                input={"matrix": _matrix(((0, 1), (2, 3))), "max_power": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="tropical.matrix.assignment_profile.compute",
        title="Compute a tropical assignment profile",
        description="Return the extremal assignment value and every tied optimum under a strict finite permutation bound.",
        request_type=MatrixAssignmentRequest,
        result_type=AssignmentResult,
        run=compute_assignment,
        tags=("tropical", "matrix", "assignment", "exact"),
        examples=(
            OperationExample(
                name="assignment",
                description="Compute the minimum assignment of a 2 by 2 MIN_PLUS matrix; the matrix must be square.",
                input={
                    "matrix": _matrix(
                        ((0, 4), (3, 1)),
                    )
                },
            ),
        ),
    ),
)
__all__ = [
    "TOOLS",
    "compute_assignment",
    "compute_finite_power_sum",
    "compute_matrix_multiply",
    "compute_matrix_power",
    "compute_polynomial_add",
    "compute_polynomial_evaluate",
    "compute_polynomial_multiply",
    "compute_scalar_add",
    "compute_scalar_multiply",
    "compute_scalar_power",
    "compute_vector_add",
    "compute_vector_scale",
]
