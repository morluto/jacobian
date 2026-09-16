"""Quadratic-form operation declarations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.general._models import (
    MAX_COEFFICIENT_MATRIX_AXIS,
    CoefficientMatrixRequest,
    CoefficientMatrixResult,
    EvaluationRequest,
    EvaluationResult,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    coefficient_matrix,
    evaluate_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_EVALUATION_DIGITS,
    MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS,
    MAX_QUADRATIC_EVALUATION_TERM_DIGITS,
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS,
)


def evaluate_form(request: EvaluationRequest) -> EvaluationResult:
    return EvaluationResult._from_kernel(
        request,
        value=evaluate_rational_quadratic_form(request.form, request.vector),
    )


def compute_coefficient_matrix(
    request: CoefficientMatrixRequest,
) -> CoefficientMatrixResult:
    return CoefficientMatrixResult._from_kernel(
        request,
        matrix=coefficient_matrix(request.form),
    )


TOOLS = (
    MathTool(
        operation_id="quadratic_form.evaluate.compute",
        title="Evaluate an exact rational quadratic form",
        description=(
            "Evaluate Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j exactly over QQ. "
            "Axis labels are unique; diagonal_coefficients carries exactly "
            "one coefficient per axis label and every cross-term index lies "
            "within that axis. The ordered vector axis must equal the form "
            "axis. The form stores polynomial coefficients; its polar matrix "
            "has diagonal 2*a_i and off-diagonal c_ij. Admission also "
            "requires the total materialized support (diagonal coefficients "
            "plus cross terms) to stay within "
            f"{MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms, and the active "
            "monomial denominator digits d to satisfy d + "
            f"{MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + len(str(t)) <= "
            f"{MAX_QUADRATIC_EVALUATION_DIGITS}, where t is the active term "
            "count."
        ),
        request_type=EvaluationRequest,
        result_type=EvaluationResult,
        run=evaluate_form,
        tags=("algebra", "quadratic-form", "exact-rational"),
        examples=(
            OperationExample(
                name="binary_cross_term",
                description="Evaluate 2*x^2+3*x*y+5*y^2 at (1/2, 2); unique labels "
                "with exactly one entry each, cross indices on that axis, "
                "matching form/vector axes, "
                f"{MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS}/"
                f"{MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS}-digit per-entry "
                "bounds, support within "
                f"{MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms, and d + "
                f"{MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + digits(t) within "
                f"{MAX_QUADRATIC_EVALUATION_DIGITS} on active denominators.",
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "2", "den": "1"},
                            {"num": "5", "den": "1"},
                        ],
                        "cross_terms": [
                            {
                                "left": 0,
                                "right": 1,
                                "coefficient": {"num": "3", "den": "1"},
                            }
                        ],
                    },
                    "vector": {
                        "axis": ["x", "y"],
                        "coordinates": [
                            {"num": "1", "den": "2"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.coefficient_matrix.compute",
        title="Compute the symmetric coefficient matrix of a quadratic form",
        description=(
            "Return the exact symmetric matrix A with Q(x)=x^T A x for one "
            "rational quadratic form on its declared axis, which serves as "
            "both the row and column axis. Diagonal entry (i,i) is the x_i^2 "
            "polynomial coefficient and off-diagonal entries (i,j)/(j,i) are "
            f"half the x_i*x_j polynomial coefficient. Admission requires n <= "
            f"{MAX_COEFFICIENT_MATRIX_AXIS} for a form of dimension n."
        ),
        request_type=CoefficientMatrixRequest,
        result_type=CoefficientMatrixResult,
        run=compute_coefficient_matrix,
        tags=("algebra", "quadratic-form", "coefficient-matrix", "exact-rational"),
        discovery_terms=(
            "quadratic form Gram matrix",
            "symmetric matrix of a quadratic form",
            "half-polar coefficient matrix",
        ),
        examples=(
            OperationExample(
                name="binary_form_with_odd_cross_term",
                description=(
                    "Compute the symmetric matrix of 2*x^2+3*x*y+5*y^2; the "
                    "form axis must carry exactly one diagonal coefficient "
                    "per label with canonical cross-term order."
                ),
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "2", "den": "1"},
                            {"num": "5", "den": "1"},
                        ],
                        "cross_terms": [
                            {
                                "left": 0,
                                "right": 1,
                                "coefficient": {"num": "3", "den": "1"},
                            }
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
