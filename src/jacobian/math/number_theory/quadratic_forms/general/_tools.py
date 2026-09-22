"""Public declarations for exact general quadratic forms."""
# ruff: noqa: F405

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.general._models import *  # noqa: F403
from jacobian.math.number_theory.quadratic_forms.general.extra_operations import *  # noqa: F403
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    coefficient_matrix,
    evaluate_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_EVALUATION_DIGITS,
    MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS,
    MAX_QUADRATIC_EVALUATION_TERM_DIGITS,
)


def evaluate_form(request: EvaluationRequest) -> EvaluationResult:
    return EvaluationResult._from_kernel(
        request, value=evaluate_rational_quadratic_form(request.form, request.vector)
    )


def compute_coefficient_matrix(
    request: CoefficientMatrixRequest,
) -> CoefficientMatrixResult:
    return CoefficientMatrixResult._from_kernel(
        request, matrix=coefficient_matrix(request.form)
    )


def compute_signature(request: FormRequest) -> SignatureResult:
    p, n, z = quadratic_signature(request.form)
    return SignatureResult(
        form=request.form,
        positive_index=p,
        negative_index=n,
        zero_index=z,
        signature=p - n,
    )


def compute_radical(request: FormRequest) -> RadicalResult:
    rank, rad = quadratic_radical(request.form)
    return RadicalResult(form=request.form, rank=rank, radical=rad)


def compute_pullback(request: PullbackRequest) -> PullbackResult:
    result = quadratic_pullback(request.form, request.matrix, request.target_axis)
    return PullbackResult(
        source_form=request.form,
        matrix=request.matrix,
        form=result,
        source_axis=request.form.axis,
        target_axis=request.target_axis,
    )


def compute_diagonalization(request: FormRequest) -> DiagonalizationResult:
    diagonal, change = quadratic_diagonalization(request.form)
    return DiagonalizationResult._from_kernel(
        form=request.form,
        diagonal=tuple(CanonicalRational.from_fraction(v) for v in diagonal),
        change=change,
    )


def compute_modular_profile(request: ModularProfileRequest) -> ModularProfileResult:
    hist, total = modular_histogram(request.form, request.modulus)
    return ModularProfileResult(
        form=request.form, modulus=request.modulus, histogram=hist, total=total
    )


def _form_example() -> dict[str, object]:
    return {
        "axis": ["x", "y"],
        "diagonal_coefficients": [{"num": "2", "den": "1"}, {"num": "5", "den": "1"}],
        "cross_terms": [
            {"left": 0, "right": 1, "coefficient": {"num": "3", "den": "1"}}
        ],
    }


TOOLS = (
    MathTool(
        operation_id="quadratic_form.evaluate.compute",
        title="Evaluate an exact rational quadratic form",
        description=(
            "Evaluate Q exactly on an axis-matched rational vector. "
            f"Materialized support is limited to {MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms; "
            f"active denominator digits d must satisfy d + {MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + len(str(t)) <= "
            f"{MAX_QUADRATIC_EVALUATION_DIGITS}."
        ),
        request_type=EvaluationRequest,
        result_type=EvaluationResult,
        run=evaluate_form,
        tags=("quadratic-form", "exact"),
        examples=(
            OperationExample(
                name="evaluate",
                description=(
                    "Evaluate 2*x^2+3*x*y+5*y^2 at (1/2,2); form and vector axes must match. "
                    f"Support is limited to {MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms; d + {MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + digits(t) within {MAX_QUADRATIC_EVALUATION_DIGITS}."
                ),
                input={
                    "form": _form_example(),
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
        title="Compute a quadratic-form coefficient matrix",
        description="Return A with Q=x transpose A x under the half-cross-term convention.",
        request_type=CoefficientMatrixRequest,
        result_type=CoefficientMatrixResult,
        run=compute_coefficient_matrix,
        tags=("quadratic-form", "matrix", "exact"),
        examples=(
            OperationExample(
                name="matrix",
                description="Compute the exact symmetric matrix; cross terms use canonical left-right indices.",
                input={"form": _form_example()},
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.signature.compute",
        title="Compute exact quadratic-form inertia",
        description="Return the exact positive, negative, and zero indices of the rational coefficient matrix.",
        request_type=FormRequest,
        result_type=SignatureResult,
        run=compute_signature,
        tags=("quadratic-form", "signature", "exact"),
        examples=(
            OperationExample(
                name="signature",
                description="Compute the signature of a rational form; coefficients use the declared axis.",
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "-1", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.rank_radical.compute",
        title="Compute the exact radical of a quadratic form",
        description="Return rank and a rational basis of the nullspace of the coefficient matrix.",
        request_type=FormRequest,
        result_type=RadicalResult,
        run=compute_radical,
        tags=("quadratic-form", "radical", "exact"),
        examples=(
            OperationExample(
                name="radical",
                description="Compute the radical of a degenerate form; the axis remains attached even for the zero radical basis.",
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.variable_change.compute",
        title="Pull back a quadratic form",
        description="Return the exact pullback A'=M transpose A M on an explicit target axis.",
        request_type=PullbackRequest,
        result_type=PullbackResult,
        run=compute_pullback,
        tags=("quadratic-form", "pullback", "exact"),
        examples=(
            OperationExample(
                name="pullback",
                description="Pull back a form through an axis-compatible matrix; matrix rows must be the source axis.",
                input={
                    "form": _form_example(),
                    "matrix": {
                        "row_count": 2,
                        "column_count": 2,
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
                            [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        ],
                    },
                    "target_axis": ["u", "v"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.rational_diagonalization.compute",
        title="Diagonalize a rational quadratic form",
        description="Return an exact tracked congruence change and diagonal coefficients.",
        request_type=FormRequest,
        result_type=DiagonalizationResult,
        run=compute_diagonalization,
        tags=("quadratic-form", "diagonalization", "exact"),
        examples=(
            OperationExample(
                name="diagonalize",
                description="Diagonalize a rational form by congruence; the result retains the source axis and change matrix.",
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
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.modular_value_profile.compute",
        title="Compute a complete modular value histogram",
        description="Enumerate every residue vector and count each form value modulo m.",
        request_type=ModularProfileRequest,
        result_type=ModularProfileResult,
        run=compute_modular_profile,
        tags=("quadratic-form", "modular", "exact"),
        examples=(
            OperationExample(
                name="modular",
                description="Compute the complete value histogram modulo 3; the form must have integral coefficients.",
                input={
                    "form": {
                        "axis": ["x"],
                        "diagonal_coefficients": [{"num": "1", "den": "1"}],
                    },
                    "modulus": 3,
                },
            ),
        ),
    ),
)
__all__ = [
    "TOOLS",
    "compute_coefficient_matrix",
    "compute_diagonalization",
    "compute_modular_profile",
    "compute_pullback",
    "compute_radical",
    "compute_signature",
    "evaluate_form",
]
