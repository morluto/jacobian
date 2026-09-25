"""Public declarations for exact general quadratic forms."""
# ruff: noqa: F405

from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FiniteBoxProfileRequest,
    FiniteBoxProfileResult,
    FiniteGaussSumRequest,
    FiniteGaussSumResult,
    ThetaRepresentingVectorsRequest,
    ThetaRepresentingVectorsResult,
    ThetaSelectedCoefficientsRequest,
    ThetaSelectedCoefficientsResult,
)
from jacobian.math.number_theory.quadratic_forms.general._models import *  # noqa: F403
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_models import (
    QuadraticFormDirectSumRequest,
    QuadraticFormDirectSumResult,
    QuadraticFormRestrictionRequest,
    QuadraticFormRestrictionResult,
)
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_operations import (
    quadratic_form_direct_sum,
    quadratic_form_restrict_coordinates,
)
from jacobian.math.number_theory.quadratic_forms.general.extra_operations import *  # noqa: F403
from jacobian.math.number_theory.quadratic_forms.general.finite_box_operations import (
    finite_box_value_profile,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    bilinear_pairing,
    coefficient_matrix,
    evaluate_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_representing_vectors,
    theta_selected_coefficients,
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


def compute_bilinear_pairing(
    request: BilinearPairingRequest,
) -> BilinearPairingResult:
    return BilinearPairingResult._from_kernel(
        request,
        value=bilinear_pairing(request.form, request.left, request.right),
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
    require_diagonalization_budget(request.form)
    diagonal, change = quadratic_diagonalization(request.form)
    basis_axis = tuple(f"basis_{index}" for index in range(len(request.form.axis)))
    return DiagonalizationResult._from_kernel(
        form=request.form,
        diagonal=tuple(CanonicalRational.from_fraction(v) for v in diagonal),
        change=change,
        source_axis=request.form.axis,
        basis_axis=basis_axis,
    )


def compute_modular_profile(request: ModularProfileRequest) -> ModularProfileResult:
    hist, total = modular_histogram(request.form, request.modulus)
    return ModularProfileResult(
        form=request.form, modulus=request.modulus, histogram=hist, total=total
    )


def compute_finite_gauss_sum(request: FiniteGaussSumRequest) -> FiniteGaussSumResult:
    return finite_quadratic_gauss_sum(request)


def compute_direct_sum(
    request: QuadraticFormDirectSumRequest,
) -> QuadraticFormDirectSumResult:
    return quadratic_form_direct_sum(request)


def compute_coordinate_restriction(
    request: QuadraticFormRestrictionRequest,
) -> QuadraticFormRestrictionResult:
    return quadratic_form_restrict_coordinates(request)


def compute_finite_box_profile(
    request: FiniteBoxProfileRequest,
) -> FiniteBoxProfileResult:
    return finite_box_value_profile(request)


def compute_theta_selected_coefficients(
    request: ThetaSelectedCoefficientsRequest,
) -> ThetaSelectedCoefficientsResult:
    return theta_selected_coefficients(request)


def compute_theta_representing_vectors(
    request: ThetaRepresentingVectorsRequest,
) -> ThetaRepresentingVectorsResult:
    return theta_representing_vectors(request)


def _form_example() -> dict[str, object]:
    return {
        "axis": ["x", "y"],
        "diagonal_coefficients": [{"num": "2", "den": "1"}, {"num": "5", "den": "1"}],
        "cross_terms": [
            {"left": 0, "right": 1, "coefficient": {"num": "3", "den": "1"}}
        ],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
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
        description=(
            "Return D=P transpose A P with the exact source row axis and "
            "diagonal-basis column axis. Dimension, exact work, intermediate "
            "growth, and result size are admitted before elimination."
        ),
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
TOOLS = (
    *TOOLS,
    MathTool(
        operation_id="quadratic_form.representing_vectors.compute",
        title="Enumerate complete quadratic-form representation vectors",
        description=(
            "Return every integer vector x, in the exact ordered form axis, at "
            "each requested value Q(x)=n. All requested fibers are complete, "
            "including empty fibers. The positive-definite search box, exact "
            "evaluation work, vector count, and coordinate bounds are admitted "
            "before enumeration."
        ),
        request_type=ThetaRepresentingVectorsRequest,
        result_type=ThetaRepresentingVectorsResult,
        run=compute_theta_representing_vectors,
        tags=("quadratic-form", "representation-vectors", "exact"),
        examples=(
            OperationExample(
                name="sum-of-two-squares-vectors",
                description=(
                    "Return every signed and ordered representation of 2 by x^2+y^2."
                ),
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                    "indices": [0, 1, 2, 3],
                },
            ),
        ),
    ),
)
TOOLS = (
    *TOOLS,
    MathTool(
        operation_id="quadratic_form.theta_selected_coefficients.compute",
        title="Compute selected quadratic-form theta coefficients",
        description=(
            "Return the exact representation numbers r_Q(n) at strictly increasing "
            "selected indices, retaining the positive-definite integral source form. "
            "The index range and number of requested entries are bounded; a proved "
            "coordinate box, exact evaluation work, and aggregate output digits are "
            "admitted before enumeration. Intervening coefficients are not built or "
            "returned."
        ),
        request_type=ThetaSelectedCoefficientsRequest,
        result_type=ThetaSelectedCoefficientsResult,
        run=compute_theta_selected_coefficients,
        tags=("quadratic-form", "theta-series", "exact"),
        examples=(
            OperationExample(
                name="sparse-square-coefficients",
                description="Read two sparse coefficients of the unary theta series.",
                input={
                    "form": {
                        "axis": ["x"],
                        "diagonal_coefficients": [{"num": "1", "den": "1"}],
                    },
                    "indices": [0, 1_000_000],
                },
            ),
        ),
    ),
)
TOOLS = (
    *TOOLS,
    MathTool(
        operation_id="quadratic_form.bilinear_pairing.compute",
        title="Compute the polar pairing of a quadratic form",
        description=(
            "Return Q(x+y)-Q(x)-Q(y) exactly for two rational vectors on the "
            "form axis. The result uses the full polar bilinear convention; "
            "odd cross coefficients are retained without halving."
        ),
        request_type=BilinearPairingRequest,
        result_type=BilinearPairingResult,
        run=compute_bilinear_pairing,
        tags=("quadratic-form", "bilinear", "exact"),
        examples=(
            OperationExample(
                name="odd_cross_coefficient",
                description="Polarize x^2+3xy+2y^2 at (1,2) and (3,-1).",
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                        "cross_terms": [
                            {
                                "left": 0,
                                "right": 1,
                                "coefficient": {"num": "3", "den": "1"},
                            }
                        ],
                    },
                    "left": {
                        "axis": ["x", "y"],
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                        ],
                    },
                    "right": {
                        "axis": ["x", "y"],
                        "coordinates": [
                            {"num": "3", "den": "1"},
                            {"num": "-1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.direct_sum.compute",
        title="Take the orthogonal direct sum of rational quadratic forms",
        description=(
            "Combine an ordered finite family of QQ forms on disjoint generated "
            "coordinate axes. Return the exact block sum with coordinate inclusion "
            "and projection matrices; the aggregate dimension is at most 128 and "
            "the aggregate polynomial support at most 4096 terms. Under that "
            "envelope the output is at most 4,341,760 aggregate decimal digits, "
            "within the operation's 8,000,000-digit output bound."
        ),
        request_type=QuadraticFormDirectSumRequest,
        result_type=QuadraticFormDirectSumResult,
        run=compute_direct_sum,
        tags=("quadratic-form", "direct-sum", "exact"),
        examples=(
            OperationExample(
                name="orthogonal_sum_with_repeated_source_labels",
                description=(
                    "Sum two rational forms whose local labels overlap; generated "
                    "factor-coordinate axes keep the summands disjoint."
                ),
                input={
                    "forms": [
                        {
                            "axis": ["x", "y"],
                            "diagonal_coefficients": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "cross_terms": [
                                {
                                    "left": 0,
                                    "right": 1,
                                    "coefficient": {"num": "3", "den": "1"},
                                }
                            ],
                        },
                        {
                            "axis": ["x"],
                            "diagonal_coefficients": [{"num": "2", "den": "1"}],
                        },
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.coordinate_restriction.compute",
        title="Restrict a rational quadratic form to coordinate axes",
        description=(
            "Restrict a QQ form to an ordered subset of its existing coordinate "
            "axes. Return the source form, the restricted form, and the exact "
            "source-coordinate inclusion matrix. This operation handles coordinate "
            "subspaces only; it does not compute restrictions to arbitrary subspaces."
        ),
        request_type=QuadraticFormRestrictionRequest,
        result_type=QuadraticFormRestrictionResult,
        run=compute_coordinate_restriction,
        tags=("quadratic-form", "restriction", "exact"),
        examples=(
            OperationExample(
                name="ordered_coordinate_subset",
                description=(
                    "Keep coordinates y then x; the mixed term is retained and "
                    "the inclusion columns follow the requested order."
                ),
                input={
                    "form": _form_example(),
                    "selected_axis": ["y", "x"],
                },
            ),
        ),
    ),
)
TOOLS = (
    *TOOLS,
    MathTool(
        operation_id="quadratic_form.finite_box_value_profile.compute",
        title="Compute a finite-box quadratic-form value profile",
        description=(
            "For an integral rational quadratic form, count every integer vector "
            "in [-B,B]^n by its exact value. The result retains the ordered source "
            "axis and form, transports profile values as canonical exact decimal "
            "integers, and admits only complete histograms whose counts cover the "
            "declared box. The full vector count, evaluation work, and aggregate "
            "output digits are admitted before evaluating the form."
        ),
        request_type=FiniteBoxProfileRequest,
        result_type=FiniteBoxProfileResult,
        run=compute_finite_box_profile,
        tags=("quadratic-form", "finite-box", "exact"),
        examples=(
            OperationExample(
                name="indefinite-binary-form",
                description="Count values of x^2-y^2 on the complete box [-1,1]^2.",
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "-1", "den": "1"},
                        ],
                    },
                    "radius": 1,
                },
            ),
        ),
    ),
)
TOOLS = (
    *TOOLS,
    MathTool(
        operation_id="quadratic_form.finite_gauss_sum.compute",
        title="Compute an exact finite quadratic Gauss sum",
        description=(
            "Return sum_x exp(2*pi*i*Q(x)/m) over the complete residue module "
            "as an exact element of QQ[zeta_m] and its determining value histogram. "
            "Integral coefficients, modulus at most 64, and at most 2,000,000 "
            "residue vectors are required, and the retained source and canonical "
            "output are admitted by aggregate decimal digits before enumeration."
        ),
        request_type=FiniteGaussSumRequest,
        result_type=FiniteGaussSumResult,
        run=compute_finite_gauss_sum,
        tags=("quadratic-form", "gauss-sum", "cyclotomic", "exact"),
        examples=(
            OperationExample(
                name="one-variable-mod-five",
                description="The exact quadratic Gauss sum for Q(x)=x^2 modulo 5.",
                input={
                    "form": {
                        "axis": ["x"],
                        "diagonal_coefficients": [{"num": "1", "den": "1"}],
                    },
                    "modulus": 5,
                },
            ),
        ),
    ),
)


__all__ = [
    "TOOLS",
    "compute_bilinear_pairing",
    "compute_coefficient_matrix",
    "compute_coordinate_restriction",
    "compute_diagonalization",
    "compute_direct_sum",
    "compute_finite_box_profile",
    "compute_finite_gauss_sum",
    "compute_modular_profile",
    "compute_pullback",
    "compute_radical",
    "compute_signature",
    "evaluate_form",
]
